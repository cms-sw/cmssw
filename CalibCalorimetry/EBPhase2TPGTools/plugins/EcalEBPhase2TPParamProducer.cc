#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CalibCalorimetry/EBPhase2TPGTools/plugins/EcalEBPhase2TPParamProducer.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>

EcalEBPhase2TPParamProducer::EcalEBPhase2TPParamProducer(edm::ParameterSet const& pSet)
    : theBarrelGeometryToken_(esConsumes(edm::ESInputTag("", "EcalBarrel"))),
      inFile_(pSet.getUntrackedParameter<std::string>("inputFile")),
      outFile_(pSet.getUntrackedParameter<std::string>("outputFile")),
      nSamplesToUse_(pSet.getParameter<unsigned int>("nSamplesToUse")),
      useBXPlusOne_(pSet.getParameter<bool>("useBXPlusOne")),
      phaseShift_(pSet.getParameter<double>("phaseShift")),
      nWeightGroups_(pSet.getParameter<unsigned int>("nWeightGroups")),
      theEcalTPGPedestals_Token_(esConsumes(edm::ESInputTag("", ""))),
      et_sat_(pSet.getParameter<double>("Et_sat")),
      xtal_LSB_(pSet.getParameter<double>("xtal_LSB")),
      binOfMaximum_(pSet.getParameter<unsigned int>("binOfMaximum"))

{
  out_file_ = gzopen(outFile_.c_str(), "wb");

  const TString* inFileName = new TString(inFile_);
  TFile* inFile = new TFile(*inFileName, "READ");
  inFile->GetObject("average-pulse", thePulse_);
  delete inFile;
}

EcalEBPhase2TPParamProducer::~EcalEBPhase2TPParamProducer() { gzclose(out_file_); }

void EcalEBPhase2TPParamProducer::beginJob() {}

void EcalEBPhase2TPParamProducer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  using namespace edm;
  using namespace std;

  const CaloSubdetectorGeometry* theBarrelGeometry = nullptr;
  const EcalLiteDTUPedestalsMap* theEcalTPPedestals = nullptr;
  const EcalLiteDTUPedestals* peds = nullptr;

  theBarrelGeometry = &evtSetup.getData(theBarrelGeometryToken_);
  theEcalTPPedestals = &evtSetup.getData(theEcalTPGPedestals_Token_);
  std::string tmpStringConv;
  const char* tmpStringOut;

  // Compute weights  //
  std::vector<int> ampWeights[nWeightGroups_];
  std::vector<int> timeWeights[nWeightGroups_];

  for (unsigned int iGr = 0; iGr < nWeightGroups_; iGr++) {
    ampWeights[iGr] = computeWeights(nSamplesToUse_, useBXPlusOne_, phaseShift_, binOfMaximum_, 1);
    timeWeights[iGr] = computeWeights(nSamplesToUse_, useBXPlusOne_, phaseShift_, binOfMaximum_, 2);
  }

  /* write to compressed file  */
  std::stringstream toCompressStream("");
  for (unsigned int iGr = 0; iGr < nWeightGroups_; iGr++) {
    toCompressStream << "  WEIGHTAMP  " << dec << iGr << std::endl;
    for (long unsigned int i = 0; i < ampWeights[iGr].size(); i++) {
      if (ampWeights[iGr][i] < 0)
        toCompressStream << "-0x" << std::hex << abs(ampWeights[iGr][i]) << " ";
      else
        toCompressStream << "0x" << std::hex << ampWeights[iGr][i] << " ";
    }
    toCompressStream << "\n";
  }
  toCompressStream << "\n";
  tmpStringConv = toCompressStream.str();
  tmpStringOut = tmpStringConv.c_str();
  gzwrite(out_file_, tmpStringOut, std::strlen(tmpStringOut));
  toCompressStream.str(std::string());

  for (unsigned int iGr = 0; iGr < nWeightGroups_; iGr++) {
    toCompressStream << "WEIGHTTIME " << dec << iGr << std::endl;
    for (long unsigned int i = 0; i < timeWeights[iGr].size(); i++) {
      if (timeWeights[iGr][i] < 0)
        toCompressStream << "-0x" << std::hex << abs(timeWeights[iGr][i]) << " ";
      else
        toCompressStream << "0x" << std::hex << timeWeights[iGr][i] << " ";
    }
    toCompressStream << "\n";
  }

  toCompressStream << "\n";
  tmpStringConv = toCompressStream.str();
  tmpStringOut = tmpStringConv.c_str();
  gzwrite(out_file_, tmpStringOut, std::strlen(tmpStringOut));
  toCompressStream.str(std::string());

  //  fill  map between xTals and groups. If each xTal is a group there is a one-to-one map
  const std::vector<DetId>& ebCells = theBarrelGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  std::map<int, int> mapXtalToGroup;

  int iGroup = 0;
  for (const auto& it : ebCells) {
    EBDetId id(it);
    std::pair<int, int> xTalToGroup(id.rawId(), iGroup);
    mapXtalToGroup.insert(xTalToGroup);
    iGroup++;
  }

  //write to file

  for (std::map<int, int>::const_iterator it = mapXtalToGroup.begin(); it != mapXtalToGroup.end(); it++) {
    toCompressStream << "CRYSTAL " << dec << it->first << std::endl;
    toCompressStream << it->second << std::endl;
  }
  tmpStringConv = toCompressStream.str();
  tmpStringOut = tmpStringConv.c_str();
  gzwrite(out_file_, tmpStringOut, std::strlen(tmpStringOut));
  toCompressStream.str(std::string());

  /////////////////////////////////////

  for (const auto& it : ebCells) {
    EBDetId id(it);
    toCompressStream << "LINCONST " << dec << id.rawId() << std::endl;
    double theta = theBarrelGeometry->getGeometry(id)->getPosition().theta();
    EcalLiteDTUPedestalsMap::const_iterator itped = theEcalTPPedestals->getMap().find(id);

    if (itped != theEcalTPPedestals->end()) {
      peds = &(*itped);

    } else {
      edm::LogWarning("EcalEBPhase2TPParamProducer") << " could not find EcalLiteDTUPedestal entry for " << id;
    }

    int shift, mult;
    float calibCoeff = 1.;
    bool ok;
    int tmpPedByGain;
    for (unsigned int i = 0; i < ecalPh2::NGAINS; ++i) {
      ok = computeLinearizerParam(theta, gainRatio_[i], calibCoeff, shift, mult);
      if (!ok) {
        edm::LogWarning("EcalEBPhase2TPParamProducer")
            << "unable to compute the parameters for SM=" << id.ism() << " xt=" << id.ic() << " " << id.rawId();

        shift_ = 0;
        tmpPedByGain = 0;
        mult_ = 0;
        toCompressStream << " 0x0"
                         << " 0x0"
                         << " 0x0" << std::endl;
      } else {
        shift_ = shift;
        mult_ = mult;
        tmpPedByGain = (int)(peds->mean(i) + 0.5);
        toCompressStream << std::hex << " 0x" << tmpPedByGain << " 0x" << mult_ << " 0x" << shift_ << " " << i2cSub_[i]
                         << std::endl;
      }
    }
  }
  tmpStringConv = toCompressStream.str();
  tmpStringOut = tmpStringConv.c_str();
  gzwrite(out_file_, tmpStringOut, std::strlen(tmpStringOut));
  toCompressStream.str(std::string());
}

std::vector<int> EcalEBPhase2TPParamProducer::computeWeights(
    int nSamples, bool useBXPlusOne, float phaseShift, uint binOfMaximum, int type) {
  std::vector<float> sampleSet;
  std::vector<float> sampleDotSet;
  std::vector<unsigned int> clockSampleSet;
  double scaleMatrixBy = 1.;

  switch (nSamples) {
    case 12:
      clockSampleSet = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      break;
    case 8:
      switch (binOfMaximum) {
        case 8:
          clockSampleSet = {2, 3, 4, 5, 6, 7, 8, 9};
          break;
        case 6:
          clockSampleSet = {0, 1, 2, 3, 4, 5, 6, 7};
          break;
      }
      break;
    case 6:
      switch (binOfMaximum) {
        case 8:
          clockSampleSet = {3, 4, 6, 7, 8, 9};
          break;
        case 6:
          clockSampleSet = {1, 2, 4, 5, 6, 7};
          break;
      }
      break;
  }

  getPulseSampleSet(*thePulse_, phaseShift, sampleSet);
  pulseDot_ = new TGraph();
  getNumericalDeriv(*thePulse_, *pulseDot_);
  getPulseSampleSet(*pulseDot_, phaseShift, sampleDotSet);

  unsigned int fMatColumns = useBXPlusOne ? 6 : 4;

  TMatrix fMat(clockSampleSet.size(), fMatColumns);
  fillFMat(clockSampleSet, useBXPlusOne, sampleSet, sampleDotSet, fMat, binOfMaximum);
  TMatrix gMat(fMatColumns, clockSampleSet.size());

  getGMatrix(fMat, scaleMatrixBy, gMat);

  std::vector<int> tmpWeightVec;
  std::vector<int> tmpTimeWeightVec;
  unsigned int iClock = 0;
  for (unsigned int iSample = 0; iSample < 12; iSample++) {
    bool inSampleSet = false;
    for (unsigned int clockSample = 0; clockSample < clockSampleSet.size(); clockSample++) {
      if (iSample == clockSampleSet[clockSample]) {
        inSampleSet = true;
        iClock = clockSample;
        break;
      }
    }
    if (inSampleSet) {
      if (type == 1)
        tmpWeightVec.push_back(round(gMat(2, iClock) * multToInt_));  // amp weights
      if (type == 2)
        tmpWeightVec.push_back(round(gMat(3, iClock) * multToInt_));  // time weights
    } else {
      if (type == 1)
        tmpWeightVec.push_back(0);  // amp weights
      if (type == 2)
        tmpWeightVec.push_back(0);  // time weights
    }
  }

  return tmpWeightVec;
}

void EcalEBPhase2TPParamProducer::getNumericalDeriv(TGraph graph, TGraph& deriv) {
  UInt_t numPoints = graph.GetN();
  if (numPoints != NPoints_) {
    edm::LogWarning("EcalEBPhase2TPParamProducer") << "Error! Wrong amount of points in pulse graph! ";
  }
  Double_t xval;
  Double_t yval;
  Double_t xvalPOne;
  Double_t yvalPOne;

  for (UInt_t p = 0; p < NPoints_ - 1; p++) {
    graph.GetPoint(p, xval, yval);
    graph.GetPoint(p + 1, xvalPOne, yvalPOne);
    float midpoint = (xvalPOne + xval) / 2;
    float rise = yvalPOne - yval;
    float run = xvalPOne - xval;
    deriv.SetPoint(deriv.GetN(), midpoint, rise / run);
  }
  deriv.SetName("pulse_prime");
}

void EcalEBPhase2TPParamProducer::fillFMat(std::vector<UInt_t> clockSampleSet,
                                           bool useThirdPulse,
                                           std::vector<float> sampleSet,
                                           std::vector<float> sampleDotSet,
                                           TMatrix& fMat,
                                           uint binOfMaximum) {
  Int_t iShift = 8 - binOfMaximum;
  for (UInt_t i = 0; i < clockSampleSet.size(); i++) {
    Int_t tmpClockToSample = clockSampleSet[i] + iShift;
    fMat(i, 0) = sampleSet[tmpClockToSample];
    fMat(i, 1) = sampleDotSet[tmpClockToSample];
    if (tmpClockToSample > 4) {
      fMat(i, 2) = sampleSet[tmpClockToSample - 4];
      fMat(i, 3) = sampleDotSet[tmpClockToSample - 4];
    }
    if (clockSampleSet[i] > 8 && useThirdPulse) {
      fMat(i, 4) = sampleSet[tmpClockToSample - 8];
      fMat(i, 5) = sampleDotSet[tmpClockToSample - 8];
    }
  }
}

void EcalEBPhase2TPParamProducer::getGMatrix(TMatrix fMat, float scaleMatrixBy, TMatrix& gMat) {
  TMatrix FT = fMat;
  FT.T();
  TMatrix tmpFT = FT;
  TMatrix FTDotF = TMatrix(tmpFT, TMatrix::kMult, fMat);
  TMatrix InvFTDotF = FTDotF;

  //Possible for this bit to fail depending on the sample set and phase shift
  InvFTDotF.Invert();

  TMatrix tmpMat(InvFTDotF, TMatrix::kMult, FT);
  gMat = tmpMat;
  gMat *= scaleMatrixBy;
}

void EcalEBPhase2TPParamProducer::getPulseSampleSet(TGraph pulseGraph,
                                                    float phaseShift,
                                                    std::vector<float>& sampleSet) {
  for (UInt_t i = 0; i < 16; i++) {
    float t = (6.25 * i) + phaseShift;
    float y = pulseGraph.Eval(t + offset_) * norm_;
    sampleSet.push_back(y);
  }
}

bool EcalEBPhase2TPParamProducer::computeLinearizerParam(
    double theta, double gainRatio, double calibCoeff, int& shift, int& mult) {
  bool result = false;

  double factor = (16383 * (xtal_LSB_ * gainRatio * calibCoeff * sin(theta))) / et_sat_;

  //first with shift_ = 0
  //add 0.5 (for rounding) and set to int
  //Here we are getting mult with a max bit length of 8
  //and shift_ with a max bit length of 4
  mult = (int)(factor + 0.5);
  for (shift = 0; shift < 15; shift++) {
    if (mult >= 128 && mult < 256) {
      result = true;
      break;
    }
    factor *= 2;
    mult = (int)(factor + 0.5);
  }

  return result;
}

// DEfine this module as a plug-in
DEFINE_FWK_MODULE(EcalEBPhase2TPParamProducer);
