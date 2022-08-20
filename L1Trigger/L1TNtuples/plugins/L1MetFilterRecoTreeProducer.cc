// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1MetFilterRecoTreeProducer
//
/**\class L1MetFilterRecoTreeProducer L1MetFilterRecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1MetFilterRecoTreeProducer.cc

 Description: Produces tree containing reco quantities


*/

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"

// cond formats

// data formats
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"
#include <TVector2.h>

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMetFilterDataFormat.h"

//
// class declaration
//

class L1MetFilterRecoTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1MetFilterRecoTreeProducer(const edm::ParameterSet&);
  ~L1MetFilterRecoTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void doMetFilters(edm::Handle<edm::TriggerResults> trigRes, edm::TriggerNames trigNames);

public:
  L1Analysis::L1AnalysisRecoMetFilterDataFormat* metFilter_data;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  //edm::EDGetTokenT<bool>                    hbheNoiseFilterResultToken_;
  edm::EDGetTokenT<bool> badChCandFilterToken_;
  edm::EDGetTokenT<bool> badPFMuonFilterToken_;

  // debug stuff
  bool triggerResultsMissing_;
  //bool hbheNoiseFilterResultMissing_;
  bool badPFMuonFilterResultMissing_;
  bool badChCandFilterResultMissing_;
  //bool hbheNFRes = false;
  bool badPFMuonRes = false;
  bool badChCandRes = false;
};

L1MetFilterRecoTreeProducer::L1MetFilterRecoTreeProducer(const edm::ParameterSet& iConfig)
    : triggerResultsMissing_(false)
//hbheNoiseFilterResultMissing_(false)
{
  usesResource(TFileService::kSharedResource);

  triggerResultsToken_ = consumes<edm::TriggerResults>(
      iConfig.getUntrackedParameter("triggerResultsToken", edm::InputTag("TriggerResults")));

  //hbheNoiseFilterResultToken_ = consumes<bool>(iConfig.getUntrackedParameter("hbheNoiseFilterResultToken",edm::InputTag("HBHENoiseFilterResultProducer:HBHENoiseFilterResult")));

  badChCandFilterToken_ =
      consumes<bool>(iConfig.getUntrackedParameter<edm::InputTag>("badChargedCandidateFilterToken"));

  badPFMuonFilterToken_ = consumes<bool>(iConfig.getUntrackedParameter<edm::InputTag>("badPFMuonFilterToken"));

  metFilter_data = new L1Analysis::L1AnalysisRecoMetFilterDataFormat();

  // set up output
  tree_ = fs_->make<TTree>("MetFilterRecoTree", "MetFilterRecoTree");
  tree_->Branch("MetFilters", "L1Analysis::L1AnalysisRecoMetFilterDataFormat", &metFilter_data, 32000, 3);
}

L1MetFilterRecoTreeProducer::~L1MetFilterRecoTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1MetFilterRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  metFilter_data->Reset();

  //hbheNFRes = false;
  badPFMuonRes = false;
  badChCandRes = false;

  // get trigger results
  edm::Handle<edm::TriggerResults> trigRes;
  iEvent.getByToken(triggerResultsToken_, trigRes);

  if (trigRes.isValid()) {
    // get trigger names
    const edm::TriggerNames& trigNames = iEvent.triggerNames(*trigRes);
    doMetFilters(trigRes, trigNames);
  } else {
    if (!triggerResultsMissing_) {
      edm::LogWarning("MissingProduct") << "Trigger results not found.  Some MET filters will not be filled"
                                        << std::endl;
    }
    triggerResultsMissing_ = true;
  }

  // // get hbhe noise filter result
  // edm::Handle<bool> hbheNoiseFilterResult;
  // iEvent.getByToken(hbheNoiseFilterResultToken_, hbheNoiseFilterResult);

  // if(hbheNoiseFilterResult.isValid()){
  //   hbheNFRes = *hbheNoiseFilterResult;
  // }
  // else {
  //   if(!hbheNoiseFilterResultMissing_){edm::LogWarning("MissingProduct") << "HBHE Noise Filter Result not found.  Branch will not be filled" << std::endl;}
  //   hbheNoiseFilterResultMissing_=true;
  // }

  edm::Handle<bool> badChCandFilterResult;
  iEvent.getByToken(badChCandFilterToken_, badChCandFilterResult);

  if (badChCandFilterResult.isValid()) {
    badChCandRes = *badChCandFilterResult;
  } else {
    if (!badChCandFilterResultMissing_) {
      edm::LogWarning("MissingProduct")
          << "Bad Charged Hadron Candidate Filter Result not found.  Branch will not be filled" << std::endl;
    }
    badChCandFilterResultMissing_ = true;
  }

  edm::Handle<bool> badPFMuonFilterResult;
  iEvent.getByToken(badPFMuonFilterToken_, badPFMuonFilterResult);

  if (badPFMuonFilterResult.isValid()) {
    badPFMuonRes = *badPFMuonFilterResult;
  } else {
    if (!badPFMuonFilterResultMissing_) {
      edm::LogWarning("MissingProduct") << "Bad PF Muon Filter Result not found.  Branch will not be filled"
                                        << std::endl;
    }
    badPFMuonFilterResultMissing_ = true;
  }

  //metFilter_data->hbheNoiseFilter                 = hbheNFRes;
  metFilter_data->badChCandFilter = badChCandRes;
  metFilter_data->badPFMuonFilter = badPFMuonRes;

  tree_->Fill();
}

void L1MetFilterRecoTreeProducer::doMetFilters(edm::Handle<edm::TriggerResults> trigRes, edm::TriggerNames trigNames) {
  //get array size
  unsigned int numTrigs = trigNames.triggerNames().size();

  //get indices of flags from event parameter set
  unsigned int hbheNoiseFilterIndex = trigNames.triggerIndex("Flag_HBHENoiseFilter");
  unsigned int hbheNoiseIsoFilterIndex = trigNames.triggerIndex("Flag_HBHENoiseIsoFilter");
  unsigned int cscTightHalo2015FilterIndex = trigNames.triggerIndex("Flag_CSCTightHalo2015Filter");
  unsigned int globalSuperTightHalo2016FilterIndex = trigNames.triggerIndex("Flag_globalSuperTightHalo2016Filter");
  unsigned int ecalDeadCellTPFilterIndex = trigNames.triggerIndex("Flag_EcalDeadCellTriggerPrimitiveFilter");
  unsigned int goodVerticesFilterIndex = trigNames.triggerIndex("Flag_goodVertices");
  unsigned int eeBadScFilterIndex = trigNames.triggerIndex("Flag_eeBadScFilter");
  unsigned int chHadTrackResFilterIndex = trigNames.triggerIndex("Flag_chargedHadronTrackResolutionFilter");
  unsigned int muonBadTrackFilterIndex = trigNames.triggerIndex("Flag_muonBadTrackFilter");

  //set flag
  metFilter_data->hbheNoiseFilter = hbheNoiseFilterIndex < numTrigs ? trigRes->accept(hbheNoiseFilterIndex) : false;
  metFilter_data->hbheNoiseIsoFilter =
      hbheNoiseIsoFilterIndex < numTrigs ? trigRes->accept(hbheNoiseIsoFilterIndex) : false;
  metFilter_data->cscTightHalo2015Filter =
      cscTightHalo2015FilterIndex < numTrigs ? trigRes->accept(cscTightHalo2015FilterIndex) : false;
  metFilter_data->globalSuperTightHalo2016Filter =
      globalSuperTightHalo2016FilterIndex < numTrigs ? trigRes->accept(globalSuperTightHalo2016FilterIndex) : false;
  metFilter_data->ecalDeadCellTPFilter =
      ecalDeadCellTPFilterIndex < numTrigs ? trigRes->accept(ecalDeadCellTPFilterIndex) : false;
  metFilter_data->goodVerticesFilter =
      goodVerticesFilterIndex < numTrigs ? trigRes->accept(goodVerticesFilterIndex) : false;
  metFilter_data->eeBadScFilter = eeBadScFilterIndex < numTrigs ? trigRes->accept(eeBadScFilterIndex) : false;
  metFilter_data->chHadTrackResFilter =
      chHadTrackResFilterIndex < numTrigs ? trigRes->accept(chHadTrackResFilterIndex) : false;
  metFilter_data->muonBadTrackFilter =
      muonBadTrackFilterIndex < numTrigs ? trigRes->accept(muonBadTrackFilterIndex) : false;
}

// ------------ method called once each job just before starting event loop  ------------
void L1MetFilterRecoTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1MetFilterRecoTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MetFilterRecoTreeProducer);
