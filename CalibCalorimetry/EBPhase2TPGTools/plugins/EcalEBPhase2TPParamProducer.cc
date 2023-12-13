#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
//
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <TGraph.h>
#include <TFile.h>
#include <TMatrix.h>
#include <zlib.h>
/**                                                                                                                                                                            
\class EcalEBPhase2TPParamProducer                                                                                                                                          
\author L. Lutton, N. Marinelli - Univ. of Notre Dame                                                                                                                       
\brief TPG Param Builder for Phase2                                                                                                                                      
*/

class EcalEBPhase2TPParamProducer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalEBPhase2TPParamProducer(edm::ParameterSet const& pSet);
  ~EcalEBPhase2TPParamProducer() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void beginJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::vector<int> computeWeights(int type);

  void getNumericalDeriv(TGraph graph, TGraph& deriv);
  void fillFMat(std::vector<unsigned int> clockSampleSet,
                bool useThirdPulse,
                std::vector<float> sampleSet,
                std::vector<float> sampleDotSet,
                TMatrix& FMat,
                unsigned int binOfMaximum);
  void getGMatrix(TMatrix FMat, float scaleMatrixBy, TMatrix& GMat);
  void getPulseSampleSet(TGraph pulseGraph, float phaseShift, std::vector<float>& sampleSet);
  bool computeLinearizerParam(double theta, double gainRatio, double calibCoeff, int& shift, int& mult);

  const edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> theBarrelGeometryToken_;
  const edm::FileInPath inFile_;
  const std::string outFile_;
  const int nSamplesToUse_;
  const bool useBXPlusOne_;
  const double phaseShift_;
  const unsigned int nWeightGroups_;
  const edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> theEcalTPGPedestals_Token_;

  gzFile out_file_;
  TGraph* thePulse_;
  TGraph* pulseDot_;

  const UInt_t NPoints_ = 1599;  //With the CMSSW pulse

  static constexpr float norm_ = 1 / 503.109;  // with the CMSSW pulse shape
  static constexpr float offset_ = 0.;         // with the CMSSW pulse shape
  int multToInt_ = 0x1000;

  int i2cSub_[2] = {0, 0};

  const double et_sat_;
  const double xtal_LSB_;
  const unsigned int binOfMaximum_;
  static const int linTopRange_;
};

EcalEBPhase2TPParamProducer::EcalEBPhase2TPParamProducer(edm::ParameterSet const& pSet)
    : theBarrelGeometryToken_(esConsumes(edm::ESInputTag("", "EcalBarrel"))),
      inFile_(pSet.getParameter<edm::FileInPath>("inputFile")),
      outFile_(pSet.getUntrackedParameter<std::string>("outputFile")),
      nSamplesToUse_(pSet.getParameter<unsigned int>("nSamplesToUse")),
      useBXPlusOne_(pSet.getParameter<bool>("useBXPlusOne")),
      phaseShift_(pSet.getParameter<double>("phaseShift")),
      nWeightGroups_(pSet.getParameter<unsigned int>("nWeightGroups")),
      theEcalTPGPedestals_Token_(esConsumes(edm::ESInputTag("EcalLiteDTUPedestals", ""))),
      et_sat_(pSet.getParameter<double>("Et_sat")),
      xtal_LSB_(pSet.getParameter<double>("xtal_LSB")),
      binOfMaximum_(pSet.getParameter<unsigned int>("binOfMaximum"))

{
  out_file_ = gzopen(outFile_.c_str(), "wb");

  std::string filename = inFile_.fullPath();
  TFile* inFile = new TFile(filename.c_str(), "READ");

  inFile->GetObject("average-pulse", thePulse_);
  delete inFile;

  if (binOfMaximum_ != 6 && binOfMaximum_ != 8)
    edm::LogError("EcalEBPhase2TPParamProducer")
        << " Value for binOfMaximum " << binOfMaximum_ << " is wrong, The default binOfMaximum=6  will be used";

  if (nSamplesToUse_ != 6 && nSamplesToUse_ != 8 && nSamplesToUse_ != 12)
    edm::LogError("EcalEBPhase2TPParamProducer")
        << " Value for nSamplesToUse " << nSamplesToUse_ << " is wrong, The default nSamplesToUse=8 will be used";
}

EcalEBPhase2TPParamProducer::~EcalEBPhase2TPParamProducer() { gzclose(out_file_); }

void EcalEBPhase2TPParamProducer::beginJob() {}

void EcalEBPhase2TPParamProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("inputFile");
  desc.addUntracked<std::string>("outputFile");
  desc.add<unsigned int>("nSamplesToUse", 8);
  desc.add<bool>("useBXPlusOne", false);
  desc.add<double>("phaseShift", 2.581);
  desc.add<unsigned int>("nWeightGroups", 61200);
  desc.add<double>("Et_sat", 1998.36);
  desc.add<double>("xtal_LSB", 0.0488);
  desc.add<unsigned int>("binOfMaximum", 6);
  descriptions.add("ecalEBPhase2TPParamProducerDefault", desc);
}

void EcalEBPhase2TPParamProducer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  using namespace edm;
  using namespace std;

  const EcalLiteDTUPedestals* peds = nullptr;
  const auto* theBarrelGeometry = &evtSetup.getData(theBarrelGeometryToken_);
  const auto* theEcalTPPedestals = &evtSetup.getData(theEcalTPGPedestals_Token_);

  std::string tmpStringConv;
  const char* tmpStringOut;

  // Compute weights  //
  std::vector<int> ampWeights[nWeightGroups_];
  std::vector<int> timeWeights[nWeightGroups_];

  for (unsigned int iGr = 0; iGr < nWeightGroups_; iGr++) {
    ampWeights[iGr] = computeWeights(1);
    timeWeights[iGr] = computeWeights(2);
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
      edm::LogError("EcalEBPhase2TPParamProducer") << " could not find EcalLiteDTUPedestal entry for " << id;
      throw cms::Exception("could not find pedestals");
    }

    int shift, mult;
    double calibCoeff = 1.;
    bool ok;
    for (unsigned int i = 0; i < ecalPh2::NGAINS; ++i) {
      ok = computeLinearizerParam(theta, ecalph2::gains[ecalPh2::NGAINS - 1 - i], calibCoeff, shift, mult);
      if (!ok) {
        edm::LogError("EcalEBPhase2TPParamProducer")
            << "unable to compute the parameters for SM=" << id.ism() << " xt=" << id.ic() << " " << id.rawId();
        throw cms::Exception("unable to compute the parameters");

      } else {
        int tmpPedByGain = (int)(peds->mean(i) + 0.5);
        toCompressStream << std::hex << " 0x" << tmpPedByGain << " 0x" << mult << " 0x" << shift << " " << i2cSub_[i]
                         << std::endl;
      }
    }
  }
  tmpStringConv = toCompressStream.str();
  tmpStringOut = tmpStringConv.c_str();
  gzwrite(out_file_, tmpStringOut, std::strlen(tmpStringOut));
  toCompressStream.str(std::string());
}

std::vector<int> EcalEBPhase2TPParamProducer::computeWeights(int type) {
  std::vector<float> sampleSet;
  std::vector<float> sampleDotSet;
  std::vector<unsigned int> clockSampleSet;
  double scaleMatrixBy = 1.;
  int lbinOfMaximum = binOfMaximum_;

  switch (binOfMaximum_) {
    case 6:
      break;
    case 8:
      break;
    default:
      lbinOfMaximum = 6;
      break;
  }

  switch (nSamplesToUse_) {
    case 12:
      clockSampleSet = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      break;
    case 8:
      switch (lbinOfMaximum) {
        case 8:
          clockSampleSet = {2, 3, 4, 5, 6, 7, 8, 9};
          break;
        case 6:
          clockSampleSet = {0, 1, 2, 3, 4, 5, 6, 7};
          break;
      }
      break;

    case 6:
      switch (lbinOfMaximum) {
        case 8:
          clockSampleSet = {3, 4, 6, 7, 8, 9};
          break;
        case 6:
          clockSampleSet = {1, 2, 4, 5, 6, 7};
          break;
      }
      break;

    default:
      clockSampleSet = {0, 1, 2, 3, 4, 5, 6, 7};
      break;
  }

  getPulseSampleSet(*thePulse_, phaseShift_, sampleSet);
  pulseDot_ = new TGraph();
  getNumericalDeriv(*thePulse_, *pulseDot_);
  getPulseSampleSet(*pulseDot_, phaseShift_, sampleDotSet);

  unsigned int fMatColumns = useBXPlusOne_ ? 6 : 4;

  TMatrix fMat(clockSampleSet.size(), fMatColumns);
  fillFMat(clockSampleSet, useBXPlusOne_, sampleSet, sampleDotSet, fMat, lbinOfMaximum);
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
  for (UInt_t i = 0; i < ecalPh2::sampleSize; i++) {
    float t = (ecalPh2::Samp_Period * i) + phaseShift;
    float y = pulseGraph.Eval(t + offset_) * norm_;
    sampleSet.push_back(y);
  }
}

bool EcalEBPhase2TPParamProducer::computeLinearizerParam(
    double theta, double gainRatio, double calibCoeff, int& shift, int& mult) {
  bool result = false;

  static constexpr double linTopRange_ = 16383.;
  //  linTopRange_ 16383 = (2**14)-1  is setting the top of the range for the linearizer output
  double factor = (linTopRange_ * (xtal_LSB_ * gainRatio * calibCoeff * sin(theta))) / et_sat_;
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
