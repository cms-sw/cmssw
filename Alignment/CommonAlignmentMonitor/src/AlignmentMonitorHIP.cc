// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorHIP
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar 29 13:59:56 CDT 2007
// $Id$
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
// #include "PluginManager/ModuleDef.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorHIP: public AlignmentMonitorBase {
   public:
      AlignmentMonitorHIP(const edm::ParameterSet& cfg): AlignmentMonitorBase(cfg) { };
      ~AlignmentMonitorHIP() {};

      void book();
      void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      std::map<Alignable*, TH1F*> m_residuals;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

void AlignmentMonitorHIP::book() {
//    m_hist = (TH1F*)(add("/", new TH1F("hist", "hist", 10, 0.5, 10.5)));
//    m_ihist = (TH1F*)(add("/iterN/", new TH1F("ihist", "ihist", 10, 0, 1)));

   std::vector<Alignable*> alignables = pStore()->alignables();
   for (std::vector<Alignable*>::const_iterator it = alignables.begin();  it != alignables.end();  ++it) {
      char name[256], title[256];
      sprintf(name,  "xresid%d", (*it)->geomDetId().rawId());
      sprintf(title, "x track-hit for DetId %d", (*it)->geomDetId().rawId());

      m_residuals[*it] = (TH1F*)(add("/iterN/", new TH1F(name, title, 100, -5., 5.)));
   }
}

void AlignmentMonitorHIP::event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
  TrajectoryStateCombiner tsoscomb;

   for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin();  it != tracks.end();  ++it) {
      const Trajectory* traj = it->first;
//      const reco::Track* track = it->second;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	 const TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit* hit = &(*meas.recHit());
	 const DetId id = hit->geographicalId();

	 if (hit->isValid()  &&  pNavigator()->detAndSubdetInMap(id)) {
	    TrajectoryStateOnSurface tsosc = tsoscomb.combine(meas.forwardPredictedState(), meas.backwardPredictedState());

	    Alignable *alignable = pNavigator()->alignableFromDetId(id);
	    std::map<Alignable*, TH1F*>::const_iterator search = m_residuals.find(alignable);
	    while (search == m_residuals.end()  &&  (alignable = alignable->mother())) search = m_residuals.find(alignable);

	    if (search != m_residuals.end()) {
	       search->second->Fill(tsosc.localPosition().x() - hit->localPosition().x());
	    }
	 } // end if hit is valid
      } // end loop over hits
   } // end loop over tracks/trajectories
}

void AlignmentMonitorHIP::afterAlignment(const edm::EventSetup &iSetup) {
}

//
// constructors and destructor
//

// AlignmentMonitorHIP::AlignmentMonitorHIP(const AlignmentMonitorHIP& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorHIP& AlignmentMonitorHIP::operator=(const AlignmentMonitorHIP& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorHIP temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

// DEFINE_SEAL_MODULE();
// DEFINE_SEAL_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorHIP, "AlignmentMonitorHIP");
DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorHIP, "AlignmentMonitorHIP");
