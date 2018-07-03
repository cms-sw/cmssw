// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorTracksFromTrajectories
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Jun 28 01:38:33 CDT 2007
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/Utilities/interface/InputTag.h" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "TH1.h" 
#include "TObject.h" 
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <fstream>

// user include files

// 
// class definition
// 

class AlignmentMonitorTracksFromTrajectories: public AlignmentMonitorBase {
   public:
      AlignmentMonitorTracksFromTrajectories(const edm::ParameterSet& cfg);
      ~AlignmentMonitorTracksFromTrajectories() {};

      void book();
      void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment();

   private:
      MuonServiceProxy *theMuonServiceProxy;
      MuonUpdatorAtVertex *theMuonUpdatorAtVertex;
      bool m_vertexConstraint;
      edm::InputTag m_beamSpot;

      TH1F *m_diMuon_Z;
      TH1F *m_diMuon_Zforward;
      TH1F *m_diMuon_Zbarrel;
      TH1F *m_diMuon_Zbackward;
      TH1F *m_diMuon_Ups;
      TH1F *m_diMuon_Jpsi;
      TH1F *m_diMuon_log;
      TH1F *m_chi2_100;
      TH1F *m_chi2_10000;
      TH1F *m_chi2_1000000;
      TH1F *m_chi2_log;
      TH1F *m_chi2DOF_5;
      TH1F *m_chi2DOF_100;
      TH1F *m_chi2DOF_1000;
      TH1F *m_chi2DOF_log;
      TH1F *m_chi2_improvement;
      TH1F *m_chi2DOF_improvement;
      TH1F *m_pt[36];
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

// AlignmentMonitorTracksFromTrajectories::AlignmentMonitorTracksFromTrajectories(const AlignmentMonitorTracksFromTrajectories& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorTracksFromTrajectories& AlignmentMonitorTracksFromTrajectories::operator=(const AlignmentMonitorTracksFromTrajectories& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorTracksFromTrajectories temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

AlignmentMonitorTracksFromTrajectories::AlignmentMonitorTracksFromTrajectories(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorTracksFromTrajectories")
   , m_vertexConstraint(cfg.getParameter<bool>("vertexConstraint"))
   , m_beamSpot(cfg.getParameter<edm::InputTag>("beamSpot"))
{
   theMuonServiceProxy = new MuonServiceProxy(cfg.getParameter<edm::ParameterSet>("ServiceParameters"));
   theMuonUpdatorAtVertex = new MuonUpdatorAtVertex(cfg.getParameter<edm::ParameterSet>("MuonUpdatorAtVertexParameters"), theMuonServiceProxy);
}

//
// member functions
//

//////////////////////////////////////////////////////////////////////
// book()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorTracksFromTrajectories::book() {
   m_diMuon_Z = book1D("/iterN/", "diMuon_Z", "Di-muon mass (GeV)", 200, 90. - 50., 90. + 50.);
   m_diMuon_Zforward = book1D("/iterN/", "diMuon_Zforward", "Di-muon mass (GeV) eta > 1.4", 200, 90. - 50., 90. + 50.);
   m_diMuon_Zbarrel = book1D("/iterN/", "diMuon_Zbarrel", "Di-muon mass (GeV) -1.4 < eta < 1.4", 200, 90. - 50., 90. + 50.);
   m_diMuon_Zbackward = book1D("/iterN/", "diMuon_Zbackward", "Di-muon mass (GeV) eta < -1.4", 200, 90. - 50., 90. + 50.);
   m_diMuon_Ups = book1D("/iterN/", "diMuon_Ups", "Di-muon mass (GeV)", 200, 9. - 3., 9. + 3.);
   m_diMuon_Jpsi = book1D("/iterN/", "diMuon_Jpsi", "Di-muon mass (GeV)", 200, 3. - 1., 3. + 1.);
   m_diMuon_log = book1D("/iterN/", "diMuon_log", "Di-muon mass (log GeV)", 300, 0, 3);
   m_chi2_100 = book1D("/iterN/", "m_chi2_100", "Track chi^2", 100, 0, 100);
   m_chi2_10000 = book1D("/iterN/", "m_chi2_10000", "Track chi^2", 100, 0, 10000);
   m_chi2_1000000 = book1D("/iterN/", "m_chi2_1000000", "Track chi^2", 100, 1, 1000000);
   m_chi2_log = book1D("/iterN/", "m_chi2_log", "Log track chi^2", 100, -3, 7);
   m_chi2DOF_5 = book1D("/iterN/", "m_chi2DOF_5", "Track chi^2/nDOF", 100, 0, 5);
   m_chi2DOF_100 = book1D("/iterN/", "m_chi2DOF_100", "Track chi^2/nDOF", 100, 0, 100);
   m_chi2DOF_1000 = book1D("/iterN/", "m_chi2DOF_1000", "Track chi^2/nDOF", 100, 0, 1000);
   m_chi2DOF_log = book1D("/iterN/", "m_chi2DOF_log", "Log track chi^2/nDOF", 100, -3, 7);
   m_chi2_improvement = book1D("/iterN/", "m_chi2_improvement", "Track-by-track chi^2/original chi^2", 100, 0., 10.);
   m_chi2DOF_improvement = book1D("/iterN/", "m_chi2DOF_improvement", "Track-by-track (chi^2/DOF)/(original chi^2/original DOF)", 100, 0., 10.);
   for (int i = 0;  i < 36;  i++) {
      char name[100], title[100];
      snprintf(name, sizeof(name), "m_pt_phi%d", i);
      snprintf(title, sizeof(title), "Track pt (GeV) in phi bin %d/36", i);
      m_pt[i] = book1D("/iterN/", name, title, 100, 0, 100);
   }
}

//////////////////////////////////////////////////////////////////////
// event()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorTracksFromTrajectories::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
   theMuonServiceProxy->update(iSetup);

   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByLabel(m_beamSpot, beamSpot);

   GlobalVector p1, p2;
   double e1 = 0.;
   double e2 = 0.;

   for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin();  it != tracks.end();  ++it) {
      const Trajectory *traj = it->first;
      const reco::Track *track = it->second;

      int nDOF = 0;
      TrajectoryStateOnSurface closestTSOS;
      double closest = 10000.;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	 const TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit* hit = &(*meas.recHit());
//	 const DetId id = hit->geographicalId();

	 nDOF += hit->dimension();

	 TrajectoryStateOnSurface TSOS = meas.backwardPredictedState();
	 GlobalPoint where = TSOS.surface().toGlobal(LocalPoint(0,0,0));

	 if (where.mag() < closest) {
	    closest = where.mag();
	    closestTSOS = TSOS;
	 }

      } // end loop over measurements
      nDOF -= 5;

      if (closest != 10000.) {
	 std::pair<bool, FreeTrajectoryState> state;

	 if (m_vertexConstraint) {
	    state = theMuonUpdatorAtVertex->propagateWithUpdate(closestTSOS, *beamSpot);
	    // add in chi^2 contribution from vertex contratint?
	 }
	 else {
 	    state = theMuonUpdatorAtVertex->propagate(closestTSOS, *beamSpot);
	 }

	 if (state.first) {
	    double chi2 = traj->chiSquared();
	    double chi2DOF = chi2 / double(nDOF);

	    m_chi2_100->Fill(chi2);
	    m_chi2_10000->Fill(chi2);
	    m_chi2_1000000->Fill(chi2);
	    m_chi2_log->Fill(log10(chi2));

	    m_chi2DOF_5->Fill(chi2DOF);
	    m_chi2DOF_100->Fill(chi2DOF);
	    m_chi2DOF_1000->Fill(chi2DOF);
	    m_chi2DOF_log->Fill(log10(chi2DOF));
	    m_chi2_improvement->Fill(chi2 / track->chi2());
	    m_chi2DOF_improvement->Fill(chi2DOF / track->normalizedChi2());

	    GlobalVector momentum = state.second.momentum();
	    double energy = momentum.mag();

	    if (energy > e1) {
	       e2 = e1;
	       e1 = energy;
	       p2 = p1;
	       p1 = momentum;
	    }
	    else if (energy > e2) {
	       e2 = energy;
	       p2 = momentum;
	    }

	    double pt = momentum.perp();
	    double phi = momentum.phi();
	    while (phi < -M_PI) phi += 2.* M_PI;
	    while (phi > M_PI) phi -= 2.* M_PI;
	    
	    double phibin = (phi + M_PI) / (2. * M_PI) * 36.;

	    for (int i = 0;  i < 36;  i++) {
	       if (phibin < i+1) {
		  m_pt[i]->Fill(pt);
		  break;
	       }
	    }

	 } // end if propagate was successful
      } // end if we have a closest TSOS
   } // end loop over tracks

   if (e1 > 0.  &&  e2 > 0.) {
      double energy_tot = e1 + e2;
      GlobalVector momentum_tot = p1 + p2;
      double mass = sqrt(energy_tot*energy_tot - momentum_tot.mag2());
      double eta = momentum_tot.eta();

      m_diMuon_Z->Fill(mass);
      if (eta > 1.4) m_diMuon_Zforward->Fill(mass);
      else if (eta > -1.4) m_diMuon_Zbarrel->Fill(mass);
      else m_diMuon_Zbackward->Fill(mass);

      m_diMuon_Ups->Fill(mass);
      m_diMuon_Jpsi->Fill(mass);
      m_diMuon_log->Fill(log10(mass));
   } // end if we have two tracks
}

//////////////////////////////////////////////////////////////////////
// afterAlignment()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorTracksFromTrajectories::afterAlignment() {
}

//
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorTracksFromTrajectories, "AlignmentMonitorTracksFromTrajectories");
