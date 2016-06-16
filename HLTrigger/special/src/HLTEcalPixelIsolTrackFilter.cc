#include "HLTrigger/special/interface/HLTEcalPixelIsolTrackFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTEcalPixelIsolTrackFilter::HLTEcalPixelIsolTrackFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_             = iConfig.getParameter<edm::InputTag> ("candTag");
  maxEnergyIn_         = iConfig.getParameter<double> ("MaxEnergyIn");
  maxEnergyOut_        = iConfig.getParameter<double> ("MaxEnergyOut");
  nMaxTrackCandidates_ = iConfig.getParameter<int>("NMaxTrackCandidates");
  dropMultiL2Event_    = iConfig.getParameter<bool> ("DropMultiL2Event");
  candTok = consumes<reco::IsolatedPixelTrackCandidateCollection>(candTag_);
}

HLTEcalPixelIsolTrackFilter::~HLTEcalPixelIsolTrackFilter(){}

void HLTEcalPixelIsolTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag",edm::InputTag("hltIsolEcalPixelTrackProd"));
  desc.add<double>("MaxEnergyIn",1.2);
  desc.add<double>("MaxEnergyOut",1.2);
  desc.add<int>("NMaxTrackCandidates",10);
  desc.add<bool>("DropMultiL2Event",false);
  descriptions.add("isolEcalPixelTrackFilter",desc);
}

bool HLTEcalPixelIsolTrackFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recotrackcands;
  iEvent.getByToken(candTok,recotrackcands);
  if (!recotrackcands.isValid()) return false;

  int n=0;
  for (unsigned int i=0; i<recotrackcands->size(); i++) {
    edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref =
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);
    LogDebug("IsoTrk") << "candref.isNull() " << candref.isNull() << "\n";
    if (candref.isNull()) continue;
    LogDebug("IsoTrk") << "candref.track().isNull() " << candref->track().isNull() << "\n";
    if(candref->track().isNull()) continue;
    // select on transverse momentum
    LogDebug("IsoTrk") << "energyin/out: " << candref->energyIn() << "/" << candref->energyOut() << "\n";
    if (candref->energyIn()<maxEnergyIn_&& candref->energyOut()<maxEnergyOut_) {
      filterproduct.addObject(trigger::TriggerTrack, candref);
      n++;
      LogDebug("IsoTrk") << "EcalIsol:Candidate[" << n <<"] pt|eta|phi "
			 << candref->pt() << "|" << candref->eta() << "|"
			 << candref->phi() << "\n";
    }
    if(!dropMultiL2Event_ && n>=nMaxTrackCandidates_) break; 

  } 
  bool accept(n>0);
  if (dropMultiL2Event_ && n>nMaxTrackCandidates_ ) accept=false;  
  return accept;
}
	  
