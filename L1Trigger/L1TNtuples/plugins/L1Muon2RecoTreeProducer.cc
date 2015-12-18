// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1JetRecoTreeProducer
//
/**\class L1JetRecoTreeProducer L1JetRecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1JetRecoTreeProducer.cc

   Description: Produces tree containing reco quantities


*/


// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// cond formats
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

// data formats
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"

// RECO TRIGGER MATCHING:
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "TString.h"
#include "TRegexp.h"
#include <utility>

//muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

//taus
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMuon2.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMet.h"

using namespace std;

//
// class declaration
//

class L1Muon2RecoTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1Muon2RecoTreeProducer(const edm::ParameterSet&);
  ~L1Muon2RecoTreeProducer();


private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void endJob();

public:
  L1Analysis::L1AnalysisRecoMuon2*        muon;

  L1Analysis::L1AnalysisRecoMuon2DataFormat*              muon_data;

  double match_trigger(std::vector<int> &trigIndices, const trigger::TriggerObjectCollection &trigObjs, edm::Handle<trigger::TriggerEvent>  &triggerEvent, const reco::Muon &mu);
  void empty_hlt();


private:

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;

  // EDM input tags
  edm::EDGetTokenT<reco::MuonCollection>       MuonToken_;
  edm::EDGetTokenT<edm::TriggerResults>        TriggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent>      triggerSummaryLabelToken_;

  // bool triggerMatching_;
  // edm::InputTag triggerSummaryLabel_;
  // std::string triggerProcessLabel_;
  // std::vector<std::string> isoTriggerNames_;
  // std::vector<std::string> triggerNames_;

  // edm::Handle<edm::TriggerResults>     isoTriggerToken_;
  // edm::Handle<edm::TriggerResults>     TriggerToken_;

  HLTConfigProvider hltConfig_;

  // debug stuff
  unsigned int maxMuon_;
  double triggerMaxDeltaR_;
  bool triggerMatching_;
  std::string triggerProcessLabel_;
  std::vector<std::string> isoTriggerNames_;
  std::vector<std::string> triggerNames_;
  std::vector<int> isoTriggerIndices_;
  std::vector<int> triggerIndices_;

};



L1Muon2RecoTreeProducer::L1Muon2RecoTreeProducer(const edm::ParameterSet& iConfig)  
{

  maxMuon_         = iConfig.getParameter<unsigned int>("maxMuon");
  // isoTriggerToken_         = iConfig.getParameter<std::vector<std::string>>("isoTriggerNames");
  // TriggerToken_         = iConfig.getParameter<std::vector<std::string>>("triggerNames");
  MuonToken_ = consumes<reco::MuonCollection>(iConfig.getUntrackedParameter("MuonToken",edm::InputTag("muons")));

  TriggerResultsToken_      = consumes<edm::TriggerResults>(iConfig.getUntrackedParameter("TriggerResultsToken",edm::InputTag("TriggerResults")));
  triggerSummaryLabelToken_ = consumes<trigger::TriggerEvent>(iConfig.getUntrackedParameter("triggerSummaryLabelToken",edm::InputTag("hltTriggerSummaryAOD","","HLT")));
  // TriggerResultsToken_      = consumes<edm::TriggerResults>(iConfig.getUntrackedParameter("TriggerResultsToken",edm::InputTag("triggerSummaryLabel")));
  // triggerSummaryLabelToken_ = consumes<trigger::TriggerEvent>(iConfig.getUntrackedParameter("triggerSummaryLabelToken",edm::InputTag("triggerSummaryLabel")));
    //iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");

  muon           = new L1Analysis::L1AnalysisRecoMuon2();
  muon_data           = muon->getData();

  tree_=fs_->make<TTree>("Muon2RecoTree", "Muon2RecoTree");
  tree_->Branch("Muon",           "L1Analysis::L1AnalysisRecoMuon2DataFormat",         &muon_data,                32000, 3);

  triggerMaxDeltaR_    = iConfig.getParameter<double>("triggerMaxDeltaR");
  triggerMatching_     = iConfig.getUntrackedParameter<bool>("triggerMatching");
  triggerProcessLabel_ = iConfig.getUntrackedParameter<std::string>("triggerProcessLabel");
  isoTriggerNames_ = iConfig.getParameter<std::vector<std::string>>("isoTriggerNames");
  triggerNames_ = iConfig.getParameter<std::vector<std::string>>("triggerNames");

  // triggerMatching_     = iConfig.getUntrackedParameter<bool>("triggerMatching");
  // _ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  // triggerProcessLabel_ = iConfig.getUntrackedParameter<std::string>("triggerProcessLabel");
  // triggerNames_        = iConfig.getParameter<std::vector<std::string> > ("triggerNames");
  // isoTriggerNames_     = iConfig.getParameter<std::vector<std::string> > ("isoTriggerNames");
  // triggerMaxDeltaR_    = iConfig.getParameter<double> ("triggerMaxDeltaR");
  
}


L1Muon2RecoTreeProducer::~L1Muon2RecoTreeProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L1Muon2RecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  muon->Reset();
  edm::Handle<reco::MuonCollection> recoMuons;
  iEvent.getByToken(MuonToken_, recoMuons);

  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(TriggerResultsToken_, triggerResults);

  edm::Handle<trigger::TriggerEvent> triggerSummaryLabel_;
  iEvent.getByToken(triggerSummaryLabelToken_, triggerSummaryLabel_);

  if (recoMuons.isValid()) {
    muon->SetMuon(iEvent, iSetup, recoMuons, maxMuon_);
  }
  else {

  }

  int counter_mu = 0;
  
  for(reco::MuonCollection::const_iterator imu = recoMuons->begin(); 
      imu != recoMuons->end() && (unsigned) counter_mu < maxMuon_; imu++) {

    //---------------------------------------------------------------------
    // TRIGGER MATCHING:
    // if specified the reconstructed muons are matched to a trigger
    //---------------------------------------------------------------------
    if (triggerMatching_) {
      double isoMatchDeltaR = 9999.;
      double matchDeltaR = 9999.;
      int hasIsoTriggered = 0;
      int hasTriggered = 0;

      
      // first check if the trigger results are valid:
      if (triggerResults.isValid()) {      
	
	if (triggerSummaryLabel_.isValid()) {
	  // get trigger objects:
	  const trigger::TriggerObjectCollection triggerObjects = triggerSummaryLabel_->getObjects();

	  matchDeltaR = match_trigger(triggerIndices_, triggerObjects, triggerSummaryLabel_, (*imu));
	  if (matchDeltaR < triggerMaxDeltaR_)
	    hasTriggered = 1;

	  // cout<<"isoTriggerIndices_.size() = "<<isoTriggerIndices_.size()<<endl;
	  // cout<<"triggerObjects.size() = "<<triggerObjects.size()<<endl;
	  // cout<<"imu->eta() = "<<imu->eta()<<endl;

	  isoMatchDeltaR = match_trigger(isoTriggerIndices_, triggerObjects, triggerSummaryLabel_, (*imu));
	  if (isoMatchDeltaR < triggerMaxDeltaR_) 
	    hasIsoTriggered = 1;
	  
	} // end if (triggerEvent.isValid())
	
      } // end if (triggerResults.isValid())

        // fill trigger matching variables:
      muon_data->hlt_isomu.push_back(hasIsoTriggered);
      muon_data->hlt_mu.push_back(hasTriggered);
      muon_data->hlt_isoDeltaR.push_back(isoMatchDeltaR);
      muon_data->hlt_deltaR.push_back(matchDeltaR);
      
    } else {
      empty_hlt();
    } // end if (triggerMatching_)
  }
  

  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void
L1Muon2RecoTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1Muon2RecoTreeProducer::endJob() {
}

void L1Muon2RecoTreeProducer::empty_hlt(){

  muon_data->hlt_isomu.push_back(-9999);
  muon_data->hlt_mu.push_back(-9999);
  muon_data->hlt_isoDeltaR.push_back(-9999);
  muon_data->hlt_deltaR.push_back(-9999);
  
}

double L1Muon2RecoTreeProducer::match_trigger(
    std::vector<int> &trigIndices, const trigger::TriggerObjectCollection &trigObjs, 
    edm::Handle<trigger::TriggerEvent>  &triggerEvent, const reco::Muon &mu
    ) 
{
  double matchDeltaR = 9999;

  for(size_t iTrigIndex = 0; iTrigIndex < trigIndices.size(); ++iTrigIndex) {
    int triggerIndex = trigIndices[iTrigIndex];
    const std::vector<std::string> moduleLabels(hltConfig_.moduleLabels(triggerIndex));
    // find index of the last module:
    const unsigned moduleIndex = hltConfig_.size(triggerIndex)-2;
    // find index of HLT trigger name:
    const unsigned hltFilterIndex = triggerEvent->filterIndex( edm::InputTag ( moduleLabels[moduleIndex], "", triggerProcessLabel_ ) );

    if (hltFilterIndex < triggerEvent->sizeFilters()) {
      const trigger::Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
      const trigger::Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));

      const unsigned nTriggers = triggerVids.size();
      for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
        // loop over all trigger objects:
        const trigger::TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];

        double dRtmp = deltaR( mu, trigObject );

        if ( dRtmp < matchDeltaR ) {
          matchDeltaR = dRtmp;
        }

      } // loop over different trigger objects
    } // if trigger is in event (should apply hltFilter with used trigger...)
  } // loop over muon candidates

  return matchDeltaR;
}

void L1Muon2RecoTreeProducer::beginRun(const edm::Run &run, const edm::EventSetup &eventSetup) {
  // Prepare for trigger matching for each new run: 
  // Look up triggetIndices in the HLT config for the different paths
  if (triggerMatching_) {
    bool changed = true;
    if (!hltConfig_.init(run, eventSetup, triggerProcessLabel_, changed)) {
      // if you can't initialize hlt configuration, crash!
      std::cout << "Error: didn't find process" << triggerProcessLabel_ << std::endl;
      assert(false);
    }

    bool enableWildcard = true;
    for (size_t iTrig = 0; iTrig < triggerNames_.size(); ++iTrig) { 
      // prepare for regular expression (with wildcards) functionality:
      TString tNameTmp = TString(triggerNames_[iTrig]);
      TRegexp tNamePattern = TRegexp(tNameTmp, enableWildcard);
      int tIndex = -1;
      // find the trigger index:
      for (unsigned ipath = 0; ipath < hltConfig_.size(); ++ipath) {
        // use TString since it provides reg exp functionality:
        TString tmpName = TString(hltConfig_.triggerName(ipath));
        if (tmpName.Contains(tNamePattern)) {
          tIndex = int(ipath);
          triggerIndices_.push_back(tIndex);
        }
      }
      if (tIndex < 0) { // if can't find trigger path at all, give warning:
        std::cout << "Warning: Could not find trigger" << triggerNames_[iTrig] << std::endl;
        //assert(false);
      }
    } // end for triggerNames
    for (size_t iTrig = 0; iTrig < isoTriggerNames_.size(); ++iTrig) { 

      // prepare for regular expression functionality:
      TString tNameTmp = TString(isoTriggerNames_[iTrig]);
      TRegexp tNamePattern = TRegexp(tNameTmp, enableWildcard);
      int tIndex = -1;
      // find the trigger index:
      for (unsigned ipath = 0; ipath < hltConfig_.size(); ++ipath) {
        // use TString since it provides reg exp functionality:
        TString tmpName = TString(hltConfig_.triggerName(ipath));
        if (tmpName.Contains(tNamePattern)) {
          tIndex = int(ipath);
          isoTriggerIndices_.push_back(tIndex);
        }
      }
      if (tIndex < 0) { // if can't find trigger path at all, give warning:
        std::cout << "Warning: Could not find trigger" << isoTriggerNames_[iTrig] << std::endl;
        //assert(false);
      }
    } // end for isoTriggerNames
  } // end if (triggerMatching_)
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1Muon2RecoTreeProducer);
