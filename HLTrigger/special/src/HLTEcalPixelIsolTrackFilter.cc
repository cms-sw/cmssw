#include "HLTrigger/special/interface/HLTEcalPixelIsolTrackFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTEcalPixelIsolTrackFilter::HLTEcalPixelIsolTrackFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_             = iConfig.getParameter<edm::InputTag> ("candTag");
  maxEnergyIn_         = iConfig.getParameter<double> ("MaxEnergyIn");
  maxEnergyOut_        = iConfig.getParameter<double> ("MaxEnergyOut");
  nMaxTrackCandidates_ = iConfig.getParameter<int>("NMaxTrackCandidates");
  dropMultiL2Event_    = iConfig.getParameter<bool> ("DropMultiL2Event");
  candTok = consumes<reco::IsolatedPixelTrackCandidateCollection>(candTag_);
}

HLTEcalPixelIsolTrackFilter::~HLTEcalPixelIsolTrackFilter(){}

bool HLTEcalPixelIsolTrackFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {
  //  std::cout << "Inside MIP Filter" << std::endl;
  if (saveTags())
    filterproduct.addCollectionTag(candTag_);


  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recotrackcands;
  iEvent.getByToken(candTok,recotrackcands);
  if (!recotrackcands.isValid()) return false;

  //Filtering
  int n=0;
  for (unsigned int i=0; i<recotrackcands->size(); i++) {
    // Ref to Candidate object to be recorded in filter object
    edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref =
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);
    //    std::cout << "candref.isNull() " << candref.isNull() << std::endl;
    if(candref.isNull()) continue;
    //    std::cout << "candref.track().isNull() " << candref->track().isNull() << std::endl;
    if(candref->track().isNull()) continue;
    // select on transverse momentum
    //    std::cout << "energyin/out: " << candref->energyIn() << "/" << candref->energyOut() << std::endl;
    if (candref->energyIn()<maxEnergyIn_&& candref->energyOut()<maxEnergyOut_) {
      filterproduct.addObject(trigger::TriggerTrack, candref);
      n++;
    }
    // stop looping over tracks if max number is reached
    if(!dropMultiL2Event_ && n>=nMaxTrackCandidates_) break; 

  } // loop over tracks
  bool accept(n>0);
  if( dropMultiL2Event_ && n>nMaxTrackCandidates_ ) accept=false;  
  //  std::cout << "accept here" << accept << std::endl;
  return accept;
}
	  
