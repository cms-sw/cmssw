// -*- C++ -*-
//
// Package:    AlignmentMuonHIPTrajectorySelector
// Class:      AlignmentMuonHIPTrajectorySelector
// 
/**\class AlignmentMuonHIPTrajectorySelector AlignmentMuonHIPTrajectorySelector.cc Alignment/CommonAlignmentProducer/plugins/AlignmentMuonHIPTrajectorySelector.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Feb 20 10:56:46 CST 2008
// $Id: AlignmentMuonHIPTrajectorySelector.cc,v 1.2 2008/03/25 03:20:48 pivarski Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

//
// class decleration
//

class AlignmentMuonHIPTrajectorySelector : public edm::EDProducer {
   public:
      explicit AlignmentMuonHIPTrajectorySelector(const edm::ParameterSet&);
      ~AlignmentMuonHIPTrajectorySelector();

   private:
      virtual void beginJob(const edm::EventSetup&);
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      
      // ---------- member data --------------------------------
      edm::InputTag m_input;
      double m_minPt;
      double m_maxTrackerForwardRedChi2;
      int m_minTrackerDOF;

      bool m_hists;
      TH1F *m_pt, *m_tracker_forwardredchi2, *m_tracker_dof;
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
AlignmentMuonHIPTrajectorySelector::AlignmentMuonHIPTrajectorySelector(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
   , m_minPt(iConfig.getParameter<double>("minPt"))
   , m_maxTrackerForwardRedChi2(iConfig.getParameter<double>("maxTrackerForwardRedChi2"))
   , m_minTrackerDOF(iConfig.getParameter<int>("minTrackerDOF"))
   , m_hists(iConfig.getParameter<bool>("hists"))
   , m_pt(NULL), m_tracker_forwardredchi2(NULL), m_tracker_dof(NULL)
{
   produces<TrajTrackAssociationCollection>();

   if (m_hists) {
      edm::Service<TFileService> fs;
      m_pt = fs->make<TH1F>("pt", "Transverse momentum (GeV)", 100, 0., 100.);
      m_tracker_forwardredchi2 = fs->make<TH1F>("trackerForwardRedChi2", "forward-biased reduced chi2 in tracker", 100, 0., 5.);
      m_tracker_dof = fs->make<TH1F>("trackerDOF", "DOF in tracker", 61, -0.5, 60.5);
   }
}


AlignmentMuonHIPTrajectorySelector::~AlignmentMuonHIPTrajectorySelector() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
AlignmentMuonHIPTrajectorySelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // input
   edm::Handle<TrajTrackAssociationCollection> originalTrajTrackMap;
   iEvent.getByLabel(m_input, originalTrajTrackMap);

   // output
   std::auto_ptr<TrajTrackAssociationCollection> newTrajTrackMap(new TrajTrackAssociationCollection());

   for (TrajTrackAssociationCollection::const_iterator iPair = originalTrajTrackMap->begin();  iPair != originalTrajTrackMap->end();  ++iPair) {
      if (m_hists) {
	 m_pt->Fill((*(*iPair).val).pt());
      }

      if ((*(*iPair).val).pt() > m_minPt) {

	 std::vector<TrajectoryMeasurement> measurements = (*(*iPair).key).measurements();

	 double tracker_forwardchi2 = 0.;
	 double tracker_dof = 0.;
	 for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	    const TrajectoryMeasurement meas = *im;
	    const TransientTrackingRecHit* hit = &(*meas.recHit());
	    const DetId id = hit->geographicalId();

	    if (hit->isValid()  &&  id.det() == DetId::Tracker) {
	       if (hit->dimension() == 1) {
		  double residual = meas.forwardPredictedState().localPosition().x() - hit->localPosition().x();
		  double error2 = meas.forwardPredictedState().localError().positionError().xx() + hit->localPositionError().xx();

		  tracker_forwardchi2 += residual * residual / error2;
		  tracker_dof += 1.;
	       }
	       else if (hit->dimension() == 2) {
		  double residualx = meas.forwardPredictedState().localPosition().x() - hit->localPosition().x();
		  double residualy = meas.forwardPredictedState().localPosition().y() - hit->localPosition().y();
		  double errorxx2 = meas.forwardPredictedState().localError().positionError().xx() + hit->localPositionError().xx();
		  double errorxy2 = meas.forwardPredictedState().localError().positionError().xy() + hit->localPositionError().xy();
		  double erroryy2 = meas.forwardPredictedState().localError().positionError().yy() + hit->localPositionError().yy();

		  tracker_forwardchi2 += (residualx * residualx + residualy * residualy) / (errorxx2 + 2.*errorxy2 + erroryy2);
		  tracker_dof += 2.;
	       }
	    }
	 }
	 tracker_dof -= 5.;
	 double tracker_forwardredchi2 = tracker_forwardchi2 / tracker_dof;

	 if (m_hists) {
	    m_tracker_forwardredchi2->Fill(tracker_forwardredchi2);
	    m_tracker_dof->Fill(tracker_dof);
	 }

	 if (tracker_forwardredchi2 < m_maxTrackerForwardRedChi2  &&  tracker_dof >= m_minTrackerDOF) {
	    newTrajTrackMap->insert((*iPair).key, (*iPair).val);
	 } // end if passes tracker cuts
      } // end if passes pT cut
   } // end loop over original trajTrackMap

   // put it in the Event
   iEvent.put(newTrajTrackMap);
}

// ------------ method called once each job just before starting event loop  ------------
void 
AlignmentMuonHIPTrajectorySelector::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------
void 
AlignmentMuonHIPTrajectorySelector::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentMuonHIPTrajectorySelector);
