// system include files
#include <atomic>
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#define DebugLog
//
// class declaration
//

namespace AlCaHBHEMuons {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}

class AlCaHBHEMuonFilter : public edm::stream::EDFilter<edm::GlobalCache<AlCaHBHEMuons::Counters> > {
public:
  explicit AlCaHBHEMuonFilter(edm::ParameterSet const&, const AlCaHBHEMuons::Counters* count);
  ~AlCaHBHEMuonFilter();
  
  static std::unique_ptr<AlCaHBHEMuons::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::unique_ptr<AlCaHBHEMuons::Counters>(new AlCaHBHEMuons::Counters());
  }
  
  virtual bool filter(edm::Event&, edm::EventSetup const&) override;
  virtual void endStream() override;
  static  void globalEndJob(const AlCaHBHEMuons::Counters* counters);
  static  void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames_, HLTNames_;
  std::string                processName_;
  unsigned int               nRun_, nAll_, nGood_;
  edm::InputTag              triggerResults_, labelMuon_;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes_;
  edm::EDGetTokenT<reco::MuonCollection>   tok_Muon_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AlCaHBHEMuonFilter::AlCaHBHEMuonFilter(edm::ParameterSet const& iConfig, const AlCaHBHEMuons::Counters* count) :
  nRun_(0), nAll_(0), nGood_(0) {
  //now do what ever initialization is needed
  trigNames_             = iConfig.getParameter<std::vector<std::string> >("Triggers");
  processName_           = iConfig.getParameter<std::string>("ProcessName");
  triggerResults_        = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelMuon_             = iConfig.getParameter<edm::InputTag>("MuonLabel");
  
  // define tokens for access
  tok_trigRes_  = consumes<edm::TriggerResults>(triggerResults_);
  tok_Muon_     = consumes<reco::MuonCollection>(labelMuon_);
  edm::LogInfo("HcalHBHEMuon") << "Parameters read from config file \n" 
			       << "Process " << processName_;
  for (unsigned int k=0; k<trigNames_.size(); ++k)
    edm::LogInfo("HcalHBHEMuon") << "Trigger[" << k << "] " << trigNames_[k];
} // AlCaHBHEMuonFilter::AlCaHBHEMuonFilter  constructor


AlCaHBHEMuonFilter::~AlCaHBHEMuonFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaHBHEMuonFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  bool accept(false);
  ++nAll_;
#ifdef DebugLog
  edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonFilter::Run " 
			       << iEvent.id().run() << " Event " 
			       << iEvent.id().event() << " Luminosity " 
			       << iEvent.luminosityBlock() << " Bunch " 
			       << iEvent.bunchCrossing();
#endif
  //Step1: Find if the event passes one of the chosen triggers
  /////////////////////////////TriggerResults
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes_, triggerResults);
  if (triggerResults.isValid()) {
    bool ok(false);
    std::vector<std::string> modules;
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
    const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
    for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
      int hlt    = triggerResults->accept(iHLT);
      for (unsigned int i=0; i<trigNames_.size(); ++i) {
	if (triggerNames_[iHLT].find(trigNames_[i].c_str())!=std::string::npos){
	  if (hlt > 0) {
	    ok = true;
	  }
#ifdef DebugLog
	  edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonFilter::Trigger "
				       << triggerNames_[iHLT] << " Flag " 
				       << hlt << ":" << ok;
#endif
	}
      }
    }
    if (ok) {
      //Step2: Get geometry/B-field information
      //Get magnetic field
      edm::ESHandle<MagneticField> bFieldH;
      iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
      const MagneticField *bField = bFieldH.product();
      // get handles to calogeometry
      edm::ESHandle<CaloGeometry> pG;
      iSetup.get<CaloGeometryRecord>().get(pG);
      const CaloGeometry* geo = pG.product();
  
      // Relevant blocks from iEvent
      edm::Handle<reco::MuonCollection> _Muon;
      iEvent.getByToken(tok_Muon_, _Muon);
#ifdef DebugLog
      edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonFilter::Muon Handle " 
				   << _Muon.isValid();
#endif
      if (_Muon.isValid()) { 
	for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); 
	     RecMuon!= _Muon->end(); ++RecMuon)  {
#ifdef DebugLog
	  edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonFilter::Muon:Track " << RecMuon->track().isNonnull() << " innerTrack " << RecMuon->innerTrack().isNonnull() << " outerTrack " << RecMuon->outerTrack().isNonnull() << " globalTrack " << RecMuon->globalTrack().isNonnull();
#endif
	  if ((RecMuon->track().isNonnull()) &&
	      (RecMuon->innerTrack().isNonnull()) &&
	      (RecMuon->outerTrack().isNonnull()) &&
	      (RecMuon->globalTrack().isNonnull())) {
	    const reco::Track* pTrack = (RecMuon->innerTrack()).get();
	    spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, false);
#ifdef DebugLog
	    edm::LogInfo("HcalHBHEMuon")<<"AlCaHBHEMuonFilter::Propagate: ECAL "
					<< trackID.okECAL << " to HCAL " 
					<< trackID.okHCAL;
#endif
	    if ((trackID.okECAL) && (trackID.okHCAL)) {
	      accept = true;
	      break;
	    }
	  }
	}
      }
    }
  }
  // Step 4:  Return the acceptance flag
  if (accept) ++nGood_;
  return accept;

}  // AlCaHBHEMuonFilter::filter

// ------------ method called once each job just after ending the event loop  ------------
void AlCaHBHEMuonFilter::endStream() {
  globalCache()->nAll_  += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaHBHEMuonFilter::globalEndJob(const AlCaHBHEMuons::Counters* count) {
  edm::LogInfo("HcalHBHEMuon") << "Selects " << count->nGood_ << " in " 
			       << count->nAll_ << " events";
}


// ------------ method called when starting to processes a run  ------------
void AlCaHBHEMuonFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  bool flag = hltConfig_.init(iRun, iSetup, processName_, changed);
  edm::LogInfo("HcalHBHEMuon") << "Run[" << nRun_ << "] " << iRun.run() 
			       << " hltconfig.init " << flag;
}

// ------------ method called when ending the processing of a run  ------------
void AlCaHBHEMuonFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogInfo("HcalHBHEMuon") << "endRun[" << nRun_ << "] " << iRun.run();
  nRun_++;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
AlCaHBHEMuonFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaHBHEMuonFilter);
