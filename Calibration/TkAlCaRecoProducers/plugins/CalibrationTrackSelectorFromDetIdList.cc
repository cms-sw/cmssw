// -*- C++ -*-
//
// Package:    CalibrationTrackSelectorFromDetIdList
// Class:      CalibrationTrackSelectorFromDetIdList
//
/**\class CalibrationTrackSelectorFromDetIdList CalibrationTrackSelectorFromDetIdList.cc Calibration/TkAlCaRecoProducers/plugins/CalibrationTrackSelectorFromDetIdList.cc
Description: Selects tracks that have at leaast one valid hit on a given set of Tracker DetIds
*/
//
// Original Author:  Marco Musich
//         Created:  Wed Aug  22 09:17:01 CEST 2018
//
//

// system include files
#include <memory>

// user include files
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

//
// class decleration
//

class dso_hidden CalibrationTrackSelectorFromDetIdList final : public edm::stream::EDProducer<> {
public:
  explicit CalibrationTrackSelectorFromDetIdList(const edm::ParameterSet&);
  ~CalibrationTrackSelectorFromDetIdList() override;

private:
  void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<reco::TrackCollection> m_label;
  TrackCandidate makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd);

  std::vector<DetIdSelector> detidsels_;
  bool m_verbose;

  edm::ESHandle<TrackerGeometry> theGeometry;
  edm::ESHandle<MagneticField> theMagField;

};

CalibrationTrackSelectorFromDetIdList::CalibrationTrackSelectorFromDetIdList(const edm::ParameterSet& iConfig):  detidsels_() {

  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet> >("selections");
   
  for(std::vector<edm::ParameterSet>::const_iterator selconfig=selconfigs.begin();selconfig!=selconfigs.end();++selconfig) {
    DetIdSelector selection(*selconfig);
    detidsels_.push_back(selection);
  }

  m_verbose = iConfig.getUntrackedParameter<bool>("verbose");
  m_label = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("Input"));
  produces<TrackCandidateCollection>();
}


CalibrationTrackSelectorFromDetIdList::~CalibrationTrackSelectorFromDetIdList() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CalibrationTrackSelectorFromDetIdList::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  edm::Handle<std::vector<reco::Track> > tracks;
  iEvent.getByToken(m_label, tracks);
  auto output = std::make_unique<TrackCandidateCollection>();
  
  // loop on tracks
  for (std::vector<reco::Track>::const_iterator ittrk = tracks->begin(), edtrk = tracks->end(); ittrk != edtrk; ++ittrk) {
    const reco::Track *trk = &(*ittrk);

    std::vector<TrackingRecHit *> hits;
    hits.clear();
    
    bool saveTrack(false);

    for (trackingRecHit_iterator ith = trk->recHitsBegin(), edh = trk->recHitsEnd(); ith != edh; ++ith) {
      const TrackingRecHit * hit = (*ith); // ith is an iterator on edm::Ref to rechit
      DetId detid = hit->geographicalId();
      
      for (const auto &detidsel : detidsels_){
	if(detidsel.isSelected(detid)) {
	  LogDebug("CalibrationTrackSelectorFromDetIdList") << "Selected by selection " << detid;
	  saveTrack=true;
	  break;
	}
      }

      // here there will be the selection
      hits.push_back(hit->clone());

    }

    std::vector<TrackingRecHit *>::iterator begin = hits.begin(), end = hits.end();

    if(saveTrack){
      output->push_back( makeCandidate ( *ittrk, begin, end ) );
    }

  }
  iEvent.put(std::move(output));
}

TrackCandidate
CalibrationTrackSelectorFromDetIdList::makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd) {
  
  PropagationDirection   pdir = tk.seedDirection();
  PTrajectoryStateOnDet state;
  if ( pdir == anyDirection ) throw cms::Exception("UnimplementedFeature") << "Cannot work with tracks that have 'anyDirecton' \n";

  if ( (pdir == alongMomentum) == (  (tk.outerPosition()-tk.innerPosition()).Dot(tk.momentum()) >= 0 ) ) {
    // use inner state
    TrajectoryStateOnSurface originalTsosIn(trajectoryStateTransform::innerStateOnSurface(tk, *theGeometry, &*theMagField));
    state = trajectoryStateTransform::persistentState( originalTsosIn, DetId(tk.innerDetId()) );
  } else {
    // use outer state
    TrajectoryStateOnSurface originalTsosOut(trajectoryStateTransform::outerStateOnSurface(tk, *theGeometry, &*theMagField));
    state = trajectoryStateTransform::persistentState( originalTsosOut, DetId(tk.outerDetId()) );
  }
  TrajectorySeed seed(state, TrackCandidate::RecHitContainer(), pdir);
  TrackCandidate::RecHitContainer ownHits;
  ownHits.reserve(hitsEnd - hitsBegin);
  for ( ; hitsBegin != hitsEnd; ++hitsBegin) {
    ownHits.push_back( *hitsBegin );
  }

  TrackCandidate cand(ownHits, seed, state, tk.seedRef());
  
  return cand;
}

void CalibrationTrackSelectorFromDetIdList::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  if(m_verbose){
    for (const auto &detidsel : detidsels_){
      auto theDetIds = theGeometry.product()->detIds();
      for(const auto &theDet : theDetIds){
	if(detidsel.isSelected(theDet)) {
	  LogDebug("CalibrationTrackSelectorFromDetIdList") << "detid: " << theDet.rawId() << " is taken" << std::endl;
	}
      } 
    }
  }

}


//define this as a plug-in
DEFINE_FWK_MODULE(CalibrationTrackSelectorFromDetIdList);
