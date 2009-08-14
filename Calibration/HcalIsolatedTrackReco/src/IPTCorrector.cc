#include <vector>
#include <memory>
#include <algorithm>

// Class header file
#include "Calibration/HcalIsolatedTrackReco/interface/IPTCorrector.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
// Framework
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/TriggerResults.h"
///
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"

IPTCorrector::IPTCorrector(const edm::ParameterSet& config){
  
  corSource_=config.getParameter<edm::InputTag>("corTracksLabel");
  uncorSource_=config.getParameter<edm::InputTag>("filterLabel");
  assocCone_=config.getParameter<double>("associationCone");

  // Register the product
  produces< reco::IsolatedPixelTrackCandidateCollection >();

}

IPTCorrector::~IPTCorrector() {}


void IPTCorrector::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  reco::IsolatedPixelTrackCandidateCollection * trackCollection=new reco::IsolatedPixelTrackCandidateCollection;

  edm::Handle<reco::TrackCollection> corTracks;
  theEvent.getByLabel(corSource_,corTracks);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> fiCand;
  theEvent.getByLabel(uncorSource_,fiCand);

  std::vector< edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;

  fiCand->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

  int nCand=isoPixTrackRefs.size();

  //loop over input ipt

  for (int p=0; p<nCand; p++) 
    {
      double iptEta=isoPixTrackRefs[p]->track()->eta();
      double iptPhi=isoPixTrackRefs[p]->track()->phi();
  
      int ntrk=0;
      double minDR=100;
      reco::TrackCollection::const_iterator citSel;

      for (reco::TrackCollection::const_iterator cit=corTracks->begin(); cit!=corTracks->end(); cit++)
	{
	  double dR=deltaR(cit->eta(), cit->phi(), iptEta, iptPhi);
	  if (dR<minDR&&dR<assocCone_) 
	    {
	      ntrk++;
	      citSel=cit;
	    }
	}

      if (ntrk>0) 
	{
          reco::IsolatedPixelTrackCandidate newCandidate(reco::TrackRef(corTracks,citSel-corTracks->begin()), isoPixTrackRefs[p]->l1tau(),isoPixTrackRefs[p]->maxPtPxl(), isoPixTrackRefs[p]->sumPtPxl());
	  trackCollection->push_back(newCandidate);
	}
    }
  
  // put the product in the event
  std::auto_ptr< reco::IsolatedPixelTrackCandidateCollection > outCollection(trackCollection);
  theEvent.put(outCollection);


}
