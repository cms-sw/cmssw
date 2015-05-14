// system include files
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
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

//
// class declaration
//

class AlCaHBHEMuonFilter : public edm::EDFilter {
public:
  explicit AlCaHBHEMuonFilter(const edm::ParameterSet&);
  ~AlCaHBHEMuonFilter();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  
  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames, HLTNames;
  std::vector<int>           trigKount, trigPass;
  std::string                processName;
  int                        nRun, nAll, nGood;
  edm::InputTag              triggerEvent_, theTriggerResultsLabel, labelMuon_;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;
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
AlCaHBHEMuonFilter::AlCaHBHEMuonFilter(const edm::ParameterSet& iConfig) :
  nRun(0), nAll(0), nGood(0) {
  //now do what ever initialization is needed
  trigNames              = iConfig.getParameter<std::vector<std::string> >("Triggers");
  processName            = iConfig.getParameter<std::string>("ProcessName");
  triggerEvent_          = iConfig.getParameter<edm::InputTag>("TriggerEventLabel");
  theTriggerResultsLabel = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelMuon_             = iConfig.getParameter<edm::InputTag>("MuonLabel");
  
  // define tokens for access
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_Muon_     = consumes<reco::MuonCollection>(labelMuon_);
  std::vector<int> dummy(trigNames.size(),0);
  trigKount = trigPass = dummy;
  edm::LogInfo("HcalHBHEMuon") << "Parameters read from config file \n" 
			       << "Process " << processName;
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogInfo("HcalHBHEMuon") << "Trigger[" << k << "] " << trigNames[k];
} // AlCaHBHEMuonFilter::AlCaHBHEMuonFilter  constructor


AlCaHBHEMuonFilter::~AlCaHBHEMuonFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaHBHEMuonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accept(false);
  nAll++;
  LogDebug("HcalHBHEMuon") << "Run " << iEvent.id().run() << " Event " 
			   << iEvent.id().event() << " Luminosity " 
			   << iEvent.luminosityBlock() << " Bunch " 
			   << iEvent.bunchCrossing();

  //Step1: Find if the event passes one of the chosen triggers
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "Error! Can't get the product "
				    << triggerEvent_.label() ;
  } else {
    triggerEvent = *(triggerEventHandle.product());

    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);
    if (triggerResults.isValid()) {
      bool ok(false);
      std::vector<std::string> modules;
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	int hlt    = triggerResults->accept(iHLT);
	for (unsigned int i=0; i<trigNames.size(); ++i) {
          if (triggerNames_[iHLT].find(trigNames[i].c_str())!=std::string::npos) {
	    trigKount[i]++;
	    if (hlt > 0) {
	      ok = true;
	      trigPass[i]++;
	    }
	    LogDebug("HcalHBHEMuon") <<"This is the trigger we are looking for "
				     << triggerNames_[iHLT] << " Flag " << hlt 
				     << ":" << ok;
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
	LogDebug("HcalHBHEMuon") << "Muon Handle " << _Muon.isValid();

	if (_Muon.isValid()) { 
	  for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon!= _Muon->end(); ++RecMuon)  {
	    LogDebug("HcalHBHEMuon") << "Muon:Track " << RecMuon->track().isNonnull()
				     << " innerTrack " << RecMuon->innerTrack().isNonnull()
				     << " outerTrack " << RecMuon->outerTrack().isNonnull()
				     << " globalTrack " << RecMuon->globalTrack().isNonnull();
	    if ((RecMuon->track().isNonnull()) &&
		(RecMuon->innerTrack().isNonnull()) &&
		(RecMuon->outerTrack().isNonnull()) &&
		(RecMuon->globalTrack().isNonnull())) {
	      const reco::Track* pTrack = (RecMuon->innerTrack()).get();
	      spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, false);
	      LogDebug("HcalHBHEMuon") << "Propagate to ECAL " << trackID.okECAL
				       << " to HCAL " << trackID.okHCAL;
	      if ((trackID.okECAL) && (trackID.okHCAL)) {
		accept = true;
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  // Step 4:  Return the acceptance flag
  if (accept) nGood++;
  return accept;

}  // AlCaHBHEMuonFilter::filter

// ------------ method called once each job just after ending the event loop  ------------
void AlCaHBHEMuonFilter::endJob() {
  edm::LogInfo("HcalHBHEMuon") << "Selects " << nGood << " in " << nAll 
			       << " events from " << nRun << " runs";
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogInfo("HcalHBHEMuon") << "Trigger[" << k << "]: " << trigNames[k] 
				 << " Events " << trigKount[k] << " Passed " 
				 << trigPass[k];
}


// ------------ method called when starting to processes a run  ------------
void AlCaHBHEMuonFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  edm::LogInfo("HcalHBHEMuon") << "Run[" << nRun << "] " << iRun.run() 
			       << " hltconfig.init " << hltConfig_.init(iRun,iSetup,processName,changed);
}

// ------------ method called when ending the processing of a run  ------------
void AlCaHBHEMuonFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("HcalHBHEMuon") << "endRun[" << nRun << "] " << iRun.run();
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
