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
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"

IPTCorrector::IPTCorrector(const edm::ParameterSet& config) :
  tok_cor_(   consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("corTracksLabel")) ),
  tok_uncor_( consumes<trigger::TriggerFilterObjectWithRefs>(config.getParameter<edm::InputTag>("filterLabel")) ),
  assocCone_( config.getParameter<double>("associationCone") )
{  
  // register the product
  produces< reco::IsolatedPixelTrackCandidateCollection >();
}


void IPTCorrector::produce(edm::StreamID, edm::Event& theEvent, edm::EventSetup const&) const {

  reco::IsolatedPixelTrackCandidateCollection * trackCollection=new reco::IsolatedPixelTrackCandidateCollection;

  edm::Handle<reco::TrackCollection> corTracks;
  theEvent.getByToken(tok_cor_,corTracks);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> fiCand;
  theEvent.getByToken(tok_uncor_,fiCand);

  std::vector< edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;

  fiCand->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

  int nCand=isoPixTrackRefs.size();

  //loop over input ipt

  for (int p=0; p<nCand; p++) {
    double iptEta=isoPixTrackRefs[p]->track()->eta();
    double iptPhi=isoPixTrackRefs[p]->track()->phi();
  
    int ntrk=0;
    double minDR=100;
    reco::TrackCollection::const_iterator citSel;

    for (reco::TrackCollection::const_iterator cit=corTracks->begin(); cit!=corTracks->end(); cit++) {
      double dR=deltaR(cit->eta(), cit->phi(), iptEta, iptPhi);
      if (dR<minDR&&dR<assocCone_) {
	minDR=dR;
	ntrk++;
	citSel=cit;
      }
    }

    if (ntrk>0) {
      reco::IsolatedPixelTrackCandidate newCandidate(reco::TrackRef(corTracks,citSel-corTracks->begin()), isoPixTrackRefs[p]->l1tau(),isoPixTrackRefs[p]->maxPtPxl(), isoPixTrackRefs[p]->sumPtPxl());
      newCandidate.setEnergyIn(isoPixTrackRefs[p]->energyIn());
      newCandidate.setEnergyOut(isoPixTrackRefs[p]->energyOut());
      newCandidate.setNHitIn(isoPixTrackRefs[p]->nHitIn());
      newCandidate.setNHitOut(isoPixTrackRefs[p]->nHitOut());
      if (isoPixTrackRefs[p]->etaPhiEcalValid()) {
	std::pair<double,double> etaphi = (isoPixTrackRefs[p]->etaPhiEcal());
	newCandidate.setEtaPhiEcal(etaphi.first,etaphi.second);
      }
      trackCollection->push_back(newCandidate);
    }
  }
  
  // put the product in the event
  std::auto_ptr< reco::IsolatedPixelTrackCandidateCollection > outCollection(trackCollection);
  theEvent.put(outCollection);


}
