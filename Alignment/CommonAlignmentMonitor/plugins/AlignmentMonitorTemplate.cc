// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorTemplate
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar 29 13:59:56 CDT 2007
// $Id: AlignmentMonitorTemplate.cc,v 1.6 2010/02/25 00:27:56 wmtan Exp $
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "TH1.h"
#include "TObject.h"
// #include "PluginManager/ModuleDef.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// user include files

//
// class definition
//

class AlignmentMonitorTemplate : public AlignmentMonitorBase {
public:
  AlignmentMonitorTemplate(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
      : AlignmentMonitorBase(cfg, iC, "AlignmentMonitorTemplate"){};
  ~AlignmentMonitorTemplate() override{};

  void book() override;
  void event(const edm::Event& iEvent,
             const edm::EventSetup& iSetup,
             const ConstTrajTrackPairCollection& iTrajTracks) override;
  void afterAlignment() override;

private:
  TH1F *m_hist, *m_ihist, *m_otherdir, *m_otherdir2;
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

void AlignmentMonitorTemplate::book() {
  m_hist = book1D("/", "hist", "hist", 10, 0.5, 10.5);      // there's only one of these per job
  m_ihist = book1D("/iterN/", "ihist", "ihist", 10, 0, 1);  // there's a new one of these for each iteration
  // "/iterN/" is a special directory name, in which the "N" gets replaced by the current iteration number.

  m_otherdir = book1D("/otherdir/", "hist", "this is a histogram in another directory", 10, 0.5, 10.5);
  m_otherdir2 =
      book1D("/iterN/otherdir/", "hist", "here's one in another directory inside the iterN directories", 10, 0.5, 10.5);

  // This is a procedure that makes one histogram for each selected alignable, and puts them in the iterN directory.
  // This is not a constant-time lookup.  If you need something faster, see AlignmentMonitorMuonHIP, which has a
  // dynamically-allocated array of TH1F*s.
  const auto& alignables = pStore()->alignables();
  for (const auto& it : alignables) {
    char name[256], title[256];
    snprintf(name, sizeof(name), "xresid%d", it->geomDetId().rawId());
    snprintf(title, sizeof(title), "x track-hit for DetId %d", it->geomDetId().rawId());

    m_residuals[it] = book1D("/iterN/", name, title, 100, -5., 5.);
  }

  // Important: you create TObject pointers with the "new" operator, but YOU don't delete them.  They're deleted by the
  // base class, which knows when they are no longer needed (based on whether they are in the /iterN/ directory or not).

  // For more detail, see the twiki page:
  // https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideAlignmentMonitors    for creating a plugin like this one
  // https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideAlignmentAlgorithms#Monitoring    for configuring it
}

void AlignmentMonitorTemplate::event(const edm::Event& iEvent,
                                     const edm::EventSetup& iSetup,
                                     const ConstTrajTrackPairCollection& tracks) {
  m_hist->Fill(iteration());  // get the number of events per iteration
  m_ihist->Fill(0.5);         // get the number of events in this iteration in the central bin

  TrajectoryStateCombiner tsoscomb;

  // This is a procedure that loops over tracks/hits, calculates residuals, and fills the appropriate histogram.
  for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
    const Trajectory* traj = it->first;
    //      const reco::Track* track = it->second;
    // If your tracks are refit using the producer in RecoTracker, you'll get updated reco::Track objects with
    // each iteration, and it makes sense to make plots using these.
    // If your tracks are refit using TrackingTools/TrackRefitter, only the Trajectories will be updated.
    // We're working on that.  I'll try to remember to change this comment when the update is ready.

    std::vector<TrajectoryMeasurement> measurements = traj->measurements();
    for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin(); im != measurements.end(); ++im) {
      const TrajectoryMeasurement meas = *im;
      const TransientTrackingRecHit* hit = &(*meas.recHit());
      const DetId id = hit->geographicalId();

      if (hit->isValid() && pNavigator()->detAndSubdetInMap(id)) {
        // Combine the forward-propagated state with the backward-propagated state to get a minimally-biased residual.
        // This is the same procedure that is used to calculate residuals in the HIP algorithm
        TrajectoryStateOnSurface tsosc = tsoscomb.combine(meas.forwardPredictedState(), meas.backwardPredictedState());

        // Search for our histogram using the Alignable* -> TH1F* map
        // The "alignable = alignable->mother()" part ascends the alignable tree, because hits are on the lowest-level
        // while our histograms may be associated with higher-level Alignables.
        Alignable* alignable = pNavigator()->alignableFromDetId(id);
        std::map<Alignable*, TH1F*>::const_iterator search = m_residuals.find(alignable);
        while (search == m_residuals.end() && (alignable = alignable->mother()))
          search = m_residuals.find(alignable);

        if (search != m_residuals.end()) {
          search->second->Fill(tsosc.localPosition().x() - hit->localPosition().x());
        }
      }  // end if hit is valid
    }    // end loop over hits
  }      // end loop over tracks/trajectories
}

void AlignmentMonitorTemplate::afterAlignment() {
  m_otherdir->Fill(
      iteration());  // this one will only get one fill per iteration, because it's called in afterAlignment()
}

//
// constructors and destructor
//

// AlignmentMonitorTemplate::AlignmentMonitorTemplate(const AlignmentMonitorTemplate& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorTemplate& AlignmentMonitorTemplate::operator=(const AlignmentMonitorTemplate& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorTemplate temp(rhs);
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

//
// DEFINE_SEAL_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorTemplate, "AlignmentMonitorTemplate");
DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorTemplate, "AlignmentMonitorTemplate");
