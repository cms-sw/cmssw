/**_________________________________________________________________
class:   CorrPCCProducer.cc

description: Computes the type 1 and type 2 corrections to the luminosity
                type 1 - first (spillover from previous BXs real clusters)
                type 2 - after (comes from real activation)

authors:Sam Higginbotham (shigginb@cern.ch) and Chris Palmer (capalmer@cern.ch) 

________________________________________________________________**/
#include <memory>
#include <string>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <map>
#include <utility>
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Luminosity/interface/LumiCorrections.h"
#include "CondFormats/DataRecord/interface/LumiCorrectionsRcd.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "TMath.h"
#include "TH1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TFile.h"

class CorrPCCProducer : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit CorrPCCProducer(const edm::ParameterSet&);
  ~CorrPCCProducer() override;

private:
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void dqmEndRun(edm::Run const& runSeg, const edm::EventSetup& iSetup) final;
  void dqmEndRunProduce(const edm::Run& runSeg, const edm::EventSetup& iSetup);
  void endJob() final;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void makeCorrectionTemplate();
  float getMaximum(std::vector<float>);
  void estimateType1Frac(std::vector<float>, float&);
  void evaluateCorrectionResiduals(std::vector<float>);
  void calculateCorrections(std::vector<float>, std::vector<float>&, float&);
  void resetBlock();

  edm::EDGetTokenT<LumiInfo> lumiInfoToken;
  std::string pccSrc_;    //input file EDproducer module label
  std::string prodInst_;  //input file product instance

  std::vector<float> rawlumiBX_;         //new vector containing clusters per bxid
  std::vector<float> errOnLumiByBX_;     //standard error per bx
  std::vector<float> totalLumiByBX_;     //summed lumi
  std::vector<float> totalLumiByBXAvg_;  //summed lumi
  std::vector<float> events_;            //Number of events in each BX
  std::vector<float> correctionTemplate_;
  std::vector<float> correctionScaleFactors_;  //list of scale factors to apply.
  float overallCorrection_;                    //The Overall correction to the integrated luminosity

  unsigned int iBlock = 0;
  unsigned int minimumNumberOfEvents;

  std::map<unsigned int, LumiInfo*> lumiInfoMapPerLS;
  std::vector<unsigned int> lumiSections;
  std::map<std::pair<unsigned int, unsigned int>, LumiInfo*>::iterator lumiInfoMapIterator;
  std::map<std::pair<unsigned int, unsigned int>, LumiInfo*>
      lumiInfoMap;  //map to obtain iov for lumiOb corrections to the luminosity.
  std::map<std::pair<unsigned int, unsigned int>, unsigned int> lumiInfoCounter;  // number of lumiSections in this block

  TH1F* corrlumiAvg_h;
  TH1F* scaleFactorAvg_h;
  TH1F* lumiAvg_h;
  TH1F* type1FracHist;
  TH1F* type1resHist;
  TH1F* type2resHist;

  unsigned int maxLS = 3500;
  MonitorElement* Type1FracMon;
  MonitorElement* Type1ResMon;
  MonitorElement* Type2ResMon;

  TGraphErrors* type1FracGraph;
  TGraphErrors* type1resGraph;
  TGraphErrors* type2resGraph;
  TList* hlist;  //list for the clusters and corrections
  TFile* histoFile;

  float type1Frac;
  float mean_type1_residual;          //Type 1 residual
  float mean_type2_residual;          //Type 2 residual
  float mean_type1_residual_unc;      //Type 1 residual uncertainty rms
  float mean_type2_residual_unc;      //Type 2 residual uncertainty rms
  unsigned int nTrain;                //Number of bunch trains used in calc type 1 and 2 res, frac.
  unsigned int countLumi_;            //The lumisection count... the size of the lumiblock
  unsigned int approxLumiBlockSize_;  //The number of lumisections per block.
  unsigned int thisLS;                //Ending lumisection for the iov that we save with the lumiInfo object.

  double type2_a_;  //amplitude for the type 2 correction
  double type2_b_;  //decay width for the type 2 correction

  float pedestal;
  float pedestal_unc;
  TGraphErrors* pedestalGraph;

  LumiCorrections* pccCorrections;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
};

//--------------------------------------------------------------------------------------------------
CorrPCCProducer::CorrPCCProducer(const edm::ParameterSet& iConfig) {
  pccSrc_ =
      iConfig.getParameter<edm::ParameterSet>("CorrPCCProducerParameters").getParameter<std::string>("inLumiObLabel");
  prodInst_ =
      iConfig.getParameter<edm::ParameterSet>("CorrPCCProducerParameters").getParameter<std::string>("ProdInst");
  approxLumiBlockSize_ =
      iConfig.getParameter<edm::ParameterSet>("CorrPCCProducerParameters").getParameter<int>("approxLumiBlockSize");
  type2_a_ = iConfig.getParameter<edm::ParameterSet>("CorrPCCProducerParameters").getParameter<double>("type2_a");
  type2_b_ = iConfig.getParameter<edm::ParameterSet>("CorrPCCProducerParameters").getParameter<double>("type2_b");
  countLumi_ = 0;
  minimumNumberOfEvents = 1000;

  totalLumiByBX_.resize(LumiConstants::numBX);
  totalLumiByBXAvg_.resize(LumiConstants::numBX);
  events_.resize(LumiConstants::numBX);
  correctionScaleFactors_.resize(LumiConstants::numBX);
  correctionTemplate_.resize(LumiConstants::numBX);

  resetBlock();

  makeCorrectionTemplate();

  edm::InputTag inputPCCTag_(pccSrc_, prodInst_);

  lumiInfoToken = consumes<LumiInfo, edm::InLumi>(inputPCCTag_);

  histoFile = new TFile("CorrectionHisto.root", "RECREATE");

  type1FracGraph = new TGraphErrors();
  type1resGraph = new TGraphErrors();
  type2resGraph = new TGraphErrors();
  type1FracGraph->SetName("Type1Fraction");
  type1resGraph->SetName("Type1Res");
  type2resGraph->SetName("Type2Res");
  type1FracGraph->GetYaxis()->SetTitle("Type 1 Fraction");
  type1resGraph->GetYaxis()->SetTitle("Type 1 Residual");
  type2resGraph->GetYaxis()->SetTitle("Type 2 Residual");
  type1FracGraph->GetXaxis()->SetTitle("Unique LS ID");
  type1resGraph->GetXaxis()->SetTitle("Unique LS ID");
  type2resGraph->GetXaxis()->SetTitle("Unique LS ID");
  type1FracGraph->SetMarkerStyle(8);
  type1resGraph->SetMarkerStyle(8);
  type2resGraph->SetMarkerStyle(8);

  pedestalGraph = new TGraphErrors();
  pedestalGraph->SetName("Pedestal");
  pedestalGraph->GetYaxis()->SetTitle("pedestal value (counts) per lumi section");
  pedestalGraph->GetXaxis()->SetTitle("Unique LS ID");
  pedestalGraph->SetMarkerStyle(8);
}

//--------------------------------------------------------------------------------------------------
CorrPCCProducer::~CorrPCCProducer() {}

//--------------------------------------------------------------------------------------------------
// This method builds the single bunch response given an exponential function (type 2 only)
void CorrPCCProducer::makeCorrectionTemplate() {
  for (unsigned int bx = 1; bx < LumiConstants::numBX; bx++) {
    correctionTemplate_.at(bx) = type2_a_ * exp(-(float(bx) - 1) * type2_b_);
  }
}

//--------------------------------------------------------------------------------------------------
// Finds max lumi value
float CorrPCCProducer::getMaximum(std::vector<float> lumi_vector) {
  float max_lumi = 0;
  for (size_t i = 0; i < lumi_vector.size(); i++) {
    if (lumi_vector.at(i) > max_lumi)
      max_lumi = lumi_vector.at(i);
  }
  return max_lumi;
}

//--------------------------------------------------------------------------------------------------
// This method takes luminosity from the last bunch in a train and makes a comparison with
// the follow non-active bunch crossing to estimate the spill over fraction (type 1 afterglow).
void CorrPCCProducer::estimateType1Frac(std::vector<float> uncorrPCCPerBX, float& type1Frac) {
  std::vector<float> corrected_tmp_;
  for (size_t i = 0; i < uncorrPCCPerBX.size(); i++) {
    corrected_tmp_.push_back(uncorrPCCPerBX.at(i));
  }

  //Apply initial type 1 correction
  for (size_t k = 0; k < LumiConstants::numBX - 1; k++) {
    float bin_k = corrected_tmp_.at(k);
    corrected_tmp_.at(k + 1) = corrected_tmp_.at(k + 1) - type1Frac * bin_k;
  }

  //Apply type 2 correction
  for (size_t i = 0; i < LumiConstants::numBX - 1; i++) {
    for (size_t j = i + 1; j < i + LumiConstants::numBX - 1; j++) {
      float bin_i = corrected_tmp_.at(i);
      if (j < LumiConstants::numBX) {
        corrected_tmp_.at(j) = corrected_tmp_.at(j) - bin_i * correctionTemplate_.at(j - i);
      } else {
        corrected_tmp_.at(j - LumiConstants::numBX) =
            corrected_tmp_.at(j - LumiConstants::numBX) - bin_i * correctionTemplate_.at(j - i);
      }
    }
  }

  //Apply additional iteration for type 1 correction
  evaluateCorrectionResiduals(corrected_tmp_);
  type1Frac += mean_type1_residual;
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::evaluateCorrectionResiduals(std::vector<float> corrected_tmp_) {
  float lumiMax = getMaximum(corrected_tmp_);
  float threshold = lumiMax * 0.2;

  mean_type1_residual = 0;
  mean_type2_residual = 0;
  nTrain = 0;
  float lumi = 0;
  std::vector<float> afterGlow;
  TH1F type1("type1", "", 1000, -0.5, 0.5);
  TH1F type2("type2", "", 1000, -0.5, 0.5);
  for (size_t ibx = 2; ibx < LumiConstants::numBX - 5; ibx++) {
    lumi = corrected_tmp_.at(ibx);
    afterGlow.clear();
    afterGlow.push_back(corrected_tmp_.at(ibx + 1));
    afterGlow.push_back(corrected_tmp_.at(ibx + 2));

    //Where type 1 and type 2 residuals are computed
    if (lumi > threshold && afterGlow[0] < threshold && afterGlow[1] < threshold) {
      for (int index = 3; index < 6; index++) {
        float thisAfterGlow = corrected_tmp_.at(ibx + index);
        if (thisAfterGlow < threshold) {
          afterGlow.push_back(thisAfterGlow);
        } else {
          break;
        }
      }
      float thisType1 = 0;
      float thisType2 = 0;
      if (afterGlow.size() > 1) {
        int nAfter = 0;
        for (unsigned int index = 1; index < afterGlow.size(); index++) {
          thisType2 += afterGlow[index];
          type2.Fill(afterGlow[index] / lumi);
          nAfter++;
        }
        thisType2 /= nAfter;
      }
      thisType1 = (afterGlow[0] - thisType2) / lumi;

      type1.Fill(thisType1);
      nTrain += 1;
    }
  }

  mean_type1_residual = type1.GetMean();  //Calculate the mean value of the type 1 residual
  mean_type2_residual = type2.GetMean();  //Calculate the mean value of the type 2 residual
  mean_type1_residual_unc = type1.GetMeanError();
  mean_type2_residual_unc = type2.GetMeanError();

  histoFile->cd();
  type1.Write();
  type2.Write();
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::calculateCorrections(std::vector<float> uncorrected,
                                           std::vector<float>& correctionScaleFactors_,
                                           float& overallCorrection_) {
  type1Frac = 0;

  int nTrials = 4;

  for (int trial = 0; trial < nTrials; trial++) {
    estimateType1Frac(uncorrected, type1Frac);
    edm::LogInfo("INFO") << "type 1 fraction after iteration " << trial << " is  " << type1Frac;
  }

  //correction should never be negative
  type1Frac = std::max(0.0, (double)type1Frac);

  std::vector<float> corrected_tmp_;
  for (size_t i = 0; i < uncorrected.size(); i++) {
    corrected_tmp_.push_back(uncorrected.at(i));
  }

  //Apply all corrections
  for (size_t i = 0; i < LumiConstants::numBX - 1; i++) {
    // type 1 - first (spillover from previous BXs real clusters)
    float bin_i = corrected_tmp_.at(i);
    corrected_tmp_.at(i + 1) = corrected_tmp_.at(i + 1) - type1Frac * bin_i;

    // type 2 - after (comes from real activation)
    bin_i = corrected_tmp_.at(i);
    for (size_t j = i + 1; j < i + LumiConstants::numBX - 1; j++) {
      if (j < LumiConstants::numBX) {
        corrected_tmp_.at(j) = corrected_tmp_.at(j) - bin_i * correctionTemplate_.at(j - i);
      } else {
        corrected_tmp_.at(j - LumiConstants::numBX) =
            corrected_tmp_.at(j - LumiConstants::numBX) - bin_i * correctionTemplate_.at(j - i);
      }
    }
  }

  float lumiMax = getMaximum(corrected_tmp_);
  float threshold = lumiMax * 0.2;

  //here subtract the pedestal
  pedestal = 0.;
  pedestal_unc = 0.;
  int nped = 0;
  for (size_t i = 0; i < LumiConstants::numBX; i++) {
    if (corrected_tmp_.at(i) < threshold) {
      pedestal += corrected_tmp_.at(i);
      nped++;
    }
  }
  if (nped > 0) {
    pedestal_unc = sqrt(pedestal) / nped;
    pedestal = pedestal / nped;
  }
  for (size_t i = 0; i < LumiConstants::numBX; i++) {
    corrected_tmp_.at(i) = corrected_tmp_.at(i) - pedestal;
  }

  evaluateCorrectionResiduals(corrected_tmp_);

  float integral_uncorr_clusters = 0;
  float integral_corr_clusters = 0;

  //Calculate Per-BX correction factor and overall correction factor
  for (size_t ibx = 0; ibx < corrected_tmp_.size(); ibx++) {
    if (corrected_tmp_.at(ibx) > threshold) {
      integral_uncorr_clusters += uncorrected.at(ibx);
      integral_corr_clusters += corrected_tmp_.at(ibx);
    }
    if (corrected_tmp_.at(ibx) != 0.0 && uncorrected.at(ibx) != 0.0) {
      correctionScaleFactors_.at(ibx) = corrected_tmp_.at(ibx) / uncorrected.at(ibx);
    } else {
      correctionScaleFactors_.at(ibx) = 0.0;
    }
  }

  overallCorrection_ = integral_corr_clusters / integral_uncorr_clusters;
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  countLumi_++;
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  thisLS = lumiSeg.luminosityBlock();

  edm::Handle<LumiInfo> PCCHandle;
  lumiSeg.getByToken(lumiInfoToken, PCCHandle);

  const LumiInfo& inLumiOb = *(PCCHandle.product());

  errOnLumiByBX_ = inLumiOb.getErrorLumiAllBX();

  unsigned int totalEvents = 0;
  for (unsigned int bx = 0; bx < LumiConstants::numBX; bx++) {
    totalEvents += errOnLumiByBX_[bx];
  }

  if (totalEvents < minimumNumberOfEvents) {
    edm::LogInfo("INFO") << "number of events in this LS is too few " << totalEvents;
    //return;
  } else {
    edm::LogInfo("INFO") << "Skipping Lumisection " << thisLS;
  }

  lumiInfoMapPerLS[thisLS] = new LumiInfo();
  totalLumiByBX_ = inLumiOb.getInstLumiAllBX();
  events_ = inLumiOb.getErrorLumiAllBX();
  lumiInfoMapPerLS[thisLS]->setInstLumiAllBX(totalLumiByBX_);
  lumiInfoMapPerLS[thisLS]->setErrorLumiAllBX(events_);
  lumiSections.push_back(thisLS);
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::dqmEndRun(edm::Run const& runSeg, const edm::EventSetup& iSetup) {
  // TODO: why was this code not put here in the first place?
  dqmEndRunProduce(runSeg, iSetup);
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::dqmEndRunProduce(edm::Run const& runSeg, const edm::EventSetup& iSetup) {
  if (lumiSections.empty()) {
    return;
  }

  std::sort(lumiSections.begin(), lumiSections.end());

  edm::LogInfo("INFO") << "Number of Lumisections " << lumiSections.size() << " in run " << runSeg.run();

  //Determining integer number of blocks
  float nBlocks_f = float(lumiSections.size()) / approxLumiBlockSize_;
  unsigned int nBlocks = 1;
  if (nBlocks_f > 1) {
    if (nBlocks_f - lumiSections.size() / approxLumiBlockSize_ < 0.5) {
      nBlocks = lumiSections.size() / approxLumiBlockSize_;
    } else {
      nBlocks = lumiSections.size() / approxLumiBlockSize_ + 1;
    }
  }

  float nLSPerBlock = float(lumiSections.size()) / nBlocks;

  std::vector<std::pair<unsigned int, unsigned int>> lsKeys;
  lsKeys.clear();

  //Constructing nBlocks IOVs
  for (unsigned iKey = 0; iKey < nBlocks; iKey++) {
    lsKeys.push_back(std::make_pair(lumiSections[(unsigned int)(iKey * nLSPerBlock)],
                                    lumiSections[(unsigned int)((iKey + 1) * nLSPerBlock) - 1]));
  }

  lsKeys[0].first = 1;

  for (unsigned int lumiSection = 0; lumiSection < lumiSections.size(); lumiSection++) {
    thisLS = lumiSections[lumiSection];

    std::pair<unsigned int, unsigned int> lsKey;

    bool foundKey = false;

    for (unsigned iKey = 0; iKey < nBlocks; iKey++) {
      if ((thisLS >= lsKeys[iKey].first) && (thisLS <= lsKeys[iKey].second)) {
        lsKey = lsKeys[iKey];
        foundKey = true;
        break;
      }
    }

    if (!foundKey) {
      edm::LogInfo("WARNING") << "Didn't find key " << thisLS;
      continue;
    }

    if (lumiInfoMap.count(lsKey) == 0) {
      lumiInfoMap[lsKey] = new LumiInfo();
    }

    //Sum all lumi in IOV of lsKey
    totalLumiByBX_ = lumiInfoMap[lsKey]->getInstLumiAllBX();
    events_ = lumiInfoMap[lsKey]->getErrorLumiAllBX();

    rawlumiBX_ = lumiInfoMapPerLS[thisLS]->getInstLumiAllBX();
    errOnLumiByBX_ = lumiInfoMapPerLS[thisLS]->getErrorLumiAllBX();

    for (unsigned int bx = 0; bx < LumiConstants::numBX; bx++) {
      totalLumiByBX_[bx] += rawlumiBX_[bx];
      events_[bx] += errOnLumiByBX_[bx];
    }
    lumiInfoMap[lsKey]->setInstLumiAllBX(totalLumiByBX_);
    lumiInfoMap[lsKey]->setErrorLumiAllBX(events_);
    lumiInfoCounter[lsKey]++;
  }

  cond::Time_t thisIOV = 1;

  char* histname1 = new char[100];
  char* histname2 = new char[100];
  char* histname3 = new char[100];
  char* histTitle1 = new char[100];
  char* histTitle2 = new char[100];
  char* histTitle3 = new char[100];
  sprintf(histTitle1, "Type1Fraction_%d", runSeg.run());
  sprintf(histTitle2, "Type1Res_%d", runSeg.run());
  sprintf(histTitle3, "Type2Res_%d", runSeg.run());
  type1FracHist = new TH1F(histTitle1, histTitle1, 1000, -0.5, 0.5);
  type1resHist = new TH1F(histTitle2, histTitle2, 4000, -0.2, 0.2);
  type2resHist = new TH1F(histTitle3, histTitle3, 4000, -0.2, 0.2);
  delete[] histTitle1;
  delete[] histTitle2;
  delete[] histTitle3;

  for (lumiInfoMapIterator = lumiInfoMap.begin(); (lumiInfoMapIterator != lumiInfoMap.end()); ++lumiInfoMapIterator) {
    totalLumiByBX_ = lumiInfoMapIterator->second->getInstLumiAllBX();
    events_ = lumiInfoMapIterator->second->getErrorLumiAllBX();

    if (events_.empty()) {
      continue;
    }

    edm::LuminosityBlockID lu(runSeg.id().run(), edm::LuminosityBlockNumber_t(lumiInfoMapIterator->first.first));
    thisIOV = (cond::Time_t)(lu.value());

    sprintf(histname1,
            "CorrectedLumiAvg_%d_%d_%d_%d",
            runSeg.run(),
            iBlock,
            lumiInfoMapIterator->first.first,
            lumiInfoMapIterator->first.second);
    sprintf(histname2,
            "ScaleFactorsAvg_%d_%d_%d_%d",
            runSeg.run(),
            iBlock,
            lumiInfoMapIterator->first.first,
            lumiInfoMapIterator->first.second);
    sprintf(histname3,
            "RawLumiAvg_%d_%d_%d_%d",
            runSeg.run(),
            iBlock,
            lumiInfoMapIterator->first.first,
            lumiInfoMapIterator->first.second);

    corrlumiAvg_h = new TH1F(histname1, "", LumiConstants::numBX, 1, LumiConstants::numBX);
    scaleFactorAvg_h = new TH1F(histname2, "", LumiConstants::numBX, 1, LumiConstants::numBX);
    lumiAvg_h = new TH1F(histname3, "", LumiConstants::numBX, 1, LumiConstants::numBX);

    //Averaging by the number of events
    for (unsigned int i = 0; i < LumiConstants::numBX; i++) {
      if (events_.at(i) != 0) {
        totalLumiByBXAvg_[i] = totalLumiByBX_[i] / events_[i];
      } else {
        totalLumiByBXAvg_[i] = 0.0;
      }
    }

    calculateCorrections(totalLumiByBXAvg_, correctionScaleFactors_, overallCorrection_);

    for (unsigned int bx = 0; bx < LumiConstants::numBX; bx++) {
      corrlumiAvg_h->SetBinContent(bx, totalLumiByBXAvg_[bx] * correctionScaleFactors_[bx]);
      if (events_.at(bx) != 0) {
        corrlumiAvg_h->SetBinError(bx,
                                   totalLumiByBXAvg_[bx] * correctionScaleFactors_[bx] / TMath::Sqrt(events_.at(bx)));
      } else {
        corrlumiAvg_h->SetBinError(bx, 0.0);
      }

      scaleFactorAvg_h->SetBinContent(bx, correctionScaleFactors_[bx]);
      lumiAvg_h->SetBinContent(bx, totalLumiByBXAvg_[bx]);
    }

    //Writing the corrections to SQL lite file for db
    pccCorrections = new LumiCorrections();
    pccCorrections->setOverallCorrection(overallCorrection_);
    pccCorrections->setType1Fraction(type1Frac);
    pccCorrections->setType1Residual(mean_type1_residual);
    pccCorrections->setType2Residual(mean_type2_residual);
    pccCorrections->setCorrectionsBX(correctionScaleFactors_);

    if (poolDbService.isAvailable()) {
      poolDbService->writeOne<LumiCorrections>(pccCorrections, thisIOV, "LumiCorrectionsRcd");
    } else {
      throw std::runtime_error("PoolDBService required.");
    }

    delete pccCorrections;

    histoFile->cd();
    corrlumiAvg_h->Write();
    scaleFactorAvg_h->Write();
    lumiAvg_h->Write();

    delete corrlumiAvg_h;
    delete scaleFactorAvg_h;
    delete lumiAvg_h;

    type1FracHist->Fill(type1Frac);
    type1resHist->Fill(mean_type1_residual);
    type2resHist->Fill(mean_type2_residual);

    for (unsigned int ils = lumiInfoMapIterator->first.first; ils < lumiInfoMapIterator->first.second + 1; ils++) {
      if (ils > maxLS) {
        std::cout << "ils out of maxLS range!!" << std::endl;
        break;
      }
      Type1FracMon->setBinContent(ils, type1Frac);
      Type1FracMon->setBinError(ils, mean_type1_residual_unc);
      Type1ResMon->setBinContent(ils, mean_type1_residual);
      Type1ResMon->setBinError(ils, mean_type1_residual_unc);
      Type2ResMon->setBinContent(ils, mean_type2_residual);
      Type2ResMon->setBinError(ils, mean_type2_residual_unc);
    }

    type1FracGraph->SetPoint(iBlock, thisIOV + approxLumiBlockSize_ / 2.0, type1Frac);
    type1resGraph->SetPoint(iBlock, thisIOV + approxLumiBlockSize_ / 2.0, mean_type1_residual);
    type2resGraph->SetPoint(iBlock, thisIOV + approxLumiBlockSize_ / 2.0, mean_type2_residual);
    type1FracGraph->SetPointError(iBlock, approxLumiBlockSize_ / 2.0, mean_type1_residual_unc);
    type1resGraph->SetPointError(iBlock, approxLumiBlockSize_ / 2.0, mean_type1_residual_unc);
    type2resGraph->SetPointError(iBlock, approxLumiBlockSize_ / 2.0, mean_type2_residual_unc);
    pedestalGraph->SetPoint(iBlock, thisIOV + approxLumiBlockSize_ / 2.0, pedestal);
    pedestalGraph->SetPointError(iBlock, approxLumiBlockSize_ / 2.0, pedestal_unc);

    edm::LogInfo("INFO")
        << "iBlock type1Frac mean_type1_residual mean_type2_residual mean_type1_residual_unc mean_type2_residual_unc "
        << iBlock << " " << type1Frac << " " << mean_type1_residual << " " << mean_type2_residual << " "
        << mean_type1_residual_unc << " " << mean_type2_residual_unc;

    type1Frac = 0.0;
    mean_type1_residual = 0.0;
    mean_type2_residual = 0.0;
    mean_type1_residual_unc = 0;
    mean_type2_residual_unc = 0;
    pedestal = 0.;
    pedestal_unc = 0.;

    iBlock++;

    resetBlock();
  }
  histoFile->cd();
  type1FracHist->Write();
  type1resHist->Write();
  type2resHist->Write();

  delete type1FracHist;
  delete type1resHist;
  delete type2resHist;

  delete[] histname1;
  delete[] histname2;
  delete[] histname3;

  for (lumiInfoMapIterator = lumiInfoMap.begin(); (lumiInfoMapIterator != lumiInfoMap.end()); ++lumiInfoMapIterator) {
    delete lumiInfoMapIterator->second;
  }
  for (unsigned int lumiSection = 0; lumiSection < lumiSections.size(); lumiSection++) {
    thisLS = lumiSections[lumiSection];
    delete lumiInfoMapPerLS[thisLS];
  }
  lumiInfoMap.clear();
  lumiInfoMapPerLS.clear();
  lumiSections.clear();
  lumiInfoCounter.clear();
}

//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::resetBlock() {
  for (unsigned int bx = 0; bx < LumiConstants::numBX; bx++) {
    totalLumiByBX_[bx] = 0;
    totalLumiByBXAvg_[bx] = 0;
    events_[bx] = 0;
    correctionScaleFactors_[bx] = 1.0;
  }
}
//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& context) {
  ibooker.setCurrentFolder("AlCaReco/LumiPCC/");
  auto scope = DQMStore::IBooker::UseRunScope(ibooker);
  Type1FracMon = ibooker.book1D("type1Fraction", "Type1Fraction;Lumisection;Type 1 Fraction", maxLS, 0, maxLS);
  Type1ResMon = ibooker.book1D("type1Residual", "Type1Residual;Lumisection;Type 1 Residual", maxLS, 0, maxLS);
  Type2ResMon = ibooker.book1D("type2Residual", "Type2Residual;Lumisection;Type 2 Residual", maxLS, 0, maxLS);
}
//--------------------------------------------------------------------------------------------------
void CorrPCCProducer::endJob() {
  histoFile->cd();
  type1FracGraph->Write();
  type1resGraph->Write();
  type2resGraph->Write();
  pedestalGraph->Write();
  histoFile->Write();
  histoFile->Close();
  delete type1FracGraph;
  delete type1resGraph;
  delete type2resGraph;
  delete pedestalGraph;
}

DEFINE_FWK_MODULE(CorrPCCProducer);
