// -*- C++ -*-
//
// Package:    OverlapProblemTSOSPositionFilter
// Class:      OverlapProblemTSOSPositionFilter
//
/**\class OverlapProblemTSOSPositionFilter OverlapProblemTSOSPositionFilter.cc DebugTools/OverlapProblem/plugins/OverlapProblemTSOSPositionFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
// $Id: OverlapProblemTSOSPositionFilter.cc,v 1.1 2012/03/12 14:46:20 venturia Exp $
//
//

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TH1F.h"
//
// class decleration
//

class OverlapProblemTSOSPositionFilter : public edm::EDFilter {
public:
  explicit OverlapProblemTSOSPositionFilter(const edm::ParameterSet&);
  ~OverlapProblemTSOSPositionFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  const bool m_validOnly;
  edm::EDGetTokenT<TrajTrackAssociationCollection> m_ttacollToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
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
OverlapProblemTSOSPositionFilter::OverlapProblemTSOSPositionFilter(const edm::ParameterSet& iConfig)
    : m_validOnly(iConfig.getParameter<bool>("onlyValidRecHit")),
      m_ttacollToken(
          consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajTrackAssoCollection"))),
      m_tTopoToken(esConsumes())

{
  //now do what ever initialization is needed
}

OverlapProblemTSOSPositionFilter::~OverlapProblemTSOSPositionFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
bool OverlapProblemTSOSPositionFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // loop on trajectories and plot TSOS local coordinate

  TrajectoryStateCombiner tsoscomb;

  // Trajectory Handle

  Handle<TrajTrackAssociationCollection> ttac;
  iEvent.getByToken(m_ttacollToken, ttac);

  const auto& tTopo = iSetup.getData(m_tTopoToken);

  for (TrajTrackAssociationCollection::const_iterator pair = ttac->begin(); pair != ttac->end(); ++pair) {
    const edm::Ref<std::vector<Trajectory> >& traj = pair->key;
    //    const reco::TrackRef & trk = pair->val;
    const std::vector<TrajectoryMeasurement>& tmcoll = traj->measurements();

    for (std::vector<TrajectoryMeasurement>::const_iterator meas = tmcoll.begin(); meas != tmcoll.end(); ++meas) {
      if (!meas->updatedState().isValid())
        continue;

      TrajectoryStateOnSurface tsos = tsoscomb(meas->forwardPredictedState(), meas->backwardPredictedState());
      TransientTrackingRecHit::ConstRecHitPointer hit = meas->recHit();

      if (!hit->isValid() && m_validOnly)
        continue;

      if (hit->geographicalId().det() != DetId::Tracker)
        continue;

      if (hit->geographicalId().subdetId() != StripSubdetector::TEC)
        continue;

      if (tTopo.tecRing(hit->geographicalId()) != 6)
        continue;

      if (tsos.localPosition().y() < 6.)
        continue;

      return true;
    }
  }

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(OverlapProblemTSOSPositionFilter);
