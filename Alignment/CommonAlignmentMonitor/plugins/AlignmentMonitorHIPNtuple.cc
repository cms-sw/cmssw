// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorHIPNtuple
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Nov 12 13:30:14 CST 2007
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TTree.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorHIPNtuple: public AlignmentMonitorBase {
   public:
      AlignmentMonitorHIPNtuple(const edm::ParameterSet& cfg);
      ~AlignmentMonitorHIPNtuple() {};

      void book();
      void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      TTree *m_ntuple;
      Int_t m_ntuple_rawid;
      ULong64_t m_ntuple_time;
      Float_t m_ntuple_hitx, m_ntuple_hity, m_ntuple_resx, m_ntuple_resy;
      Float_t m_ntuple_p1, m_ntuple_p2, m_ntuple_p3, m_ntuple_p4, m_ntuple_p5, m_ntuple_p6;
      Float_t m_ntuple_e1, m_ntuple_e2, m_ntuple_e3, m_ntuple_e4, m_ntuple_e5, m_ntuple_e6;
//       Float_t m_ntuple_c12, m_ntuple_c13, m_ntuple_c14, m_ntuple_c15, m_ntuple_c16;
//       Float_t m_ntuple_c23, m_ntuple_c24, m_ntuple_c25, m_ntuple_c26;
//       Float_t m_ntuple_c34, m_ntuple_c35, m_ntuple_c36;
//       Float_t m_ntuple_c45, m_ntuple_c46;
//       Float_t m_ntuple_c56;
      Float_t m_ntuple_tanTheta;
      Int_t m_ntuple_ring, m_ntuple_chamber;

      double theMaxAllowedHitPull;
      int m_npar;
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

AlignmentMonitorHIPNtuple::AlignmentMonitorHIPNtuple(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorHIPNtuple")
   , theMaxAllowedHitPull(cfg.getParameter<double>("maxAllowedHitPull"))
   , m_npar(cfg.getParameter<int>("npar"))
{}

void AlignmentMonitorHIPNtuple::book() {
   m_ntuple = directory("/iterN/")->make<TTree>("params", "params");
   m_ntuple->Branch("rawid", &m_ntuple_rawid, "rawid/I");
   m_ntuple->Branch("time", &m_ntuple_time, "time/l");
   m_ntuple->Branch("hitx", &m_ntuple_hitx, "hitx/F");
   m_ntuple->Branch("hity", &m_ntuple_hity, "hity/F");
   m_ntuple->Branch("resx", &m_ntuple_resx, "resx/F");
   m_ntuple->Branch("resy", &m_ntuple_resy, "resy/F");

   m_ntuple->Branch("ring", &m_ntuple_ring, "ring/I");
   m_ntuple->Branch("chamber", &m_ntuple_chamber, "chamber/I");
   m_ntuple->Branch("tanTheta", &m_ntuple_tanTheta, "tanTheta/F");
   if (m_npar > 0) {
      m_ntuple->Branch("p1", &m_ntuple_p1, "p1/F");
      m_ntuple->Branch("e1", &m_ntuple_e1, "e1/F");
   }
   if (m_npar > 1) {
      m_ntuple->Branch("p2", &m_ntuple_p2, "p2/F");
      m_ntuple->Branch("e2", &m_ntuple_e2, "e2/F");
//      m_ntuple->Branch("c12", &m_ntuple_c12, "c12/F");
   }
   if (m_npar > 2) {
      m_ntuple->Branch("p3", &m_ntuple_p3, "p3/F");
      m_ntuple->Branch("e3", &m_ntuple_e3, "e3/F");
//       m_ntuple->Branch("c13", &m_ntuple_c13, "c13/F");
//       m_ntuple->Branch("c23", &m_ntuple_c23, "c23/F");
   }
   if (m_npar > 3) {
      m_ntuple->Branch("p4", &m_ntuple_p4, "p4/F");
      m_ntuple->Branch("e4", &m_ntuple_e4, "e4/F");
//       m_ntuple->Branch("c14", &m_ntuple_c14, "c14/F");
//       m_ntuple->Branch("c24", &m_ntuple_c24, "c24/F");
//       m_ntuple->Branch("c34", &m_ntuple_c34, "c34/F");
   }
   if (m_npar > 4) {
      m_ntuple->Branch("p5", &m_ntuple_p5, "p5/F");
      m_ntuple->Branch("e5", &m_ntuple_e5, "e5/F");
//       m_ntuple->Branch("c15", &m_ntuple_c15, "c15/F");
//       m_ntuple->Branch("c25", &m_ntuple_c25, "c25/F");
//       m_ntuple->Branch("c35", &m_ntuple_c35, "c35/F");
//       m_ntuple->Branch("c45", &m_ntuple_c45, "c45/F");
   }
   if (m_npar > 5) {
      m_ntuple->Branch("p6", &m_ntuple_p6, "p6/F");
      m_ntuple->Branch("e6", &m_ntuple_e6, "e6/F");
//       m_ntuple->Branch("c16", &m_ntuple_c16, "c16/F");
//       m_ntuple->Branch("c26", &m_ntuple_c26, "c26/F");
//       m_ntuple->Branch("c36", &m_ntuple_c36, "c36/F");
//       m_ntuple->Branch("c46", &m_ntuple_c46, "c46/F");
//       m_ntuple->Branch("c56", &m_ntuple_c56, "c56/F");
   }
}

void AlignmentMonitorHIPNtuple::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
   TrajectoryStateCombiner tsoscomb;
   AlignableNavigator *theAlignableDetAccessor = pNavigator();
   AlignmentParameterStore *theAlignmentParameterStore = pStore();

   for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin();  it != tracks.end();  ++it) {
      const Trajectory* traj = (*it).first;
//      const reco::Track* track = (*it).second;

      std::vector<const TransientTrackingRecHit*> hitvec;
      std::vector<TrajectoryStateOnSurface> tsosvec;

      // loop over measurements	
      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      for (std::vector<TrajectoryMeasurement>::iterator im=measurements.begin();
	   im!=measurements.end(); im++) {
	 TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit* hit = &(*meas.recHit());
	 if (hit->isValid()  &&  theAlignableDetAccessor->detAndSubdetInMap( hit->geographicalId() )) {
	    // this is the updated state (including the current hit)
	    //TrajectoryStateOnSurface tsos=meas.updatedState();
	    // combine fwd and bwd predicted state to get state 
	    // which excludes current hit

	    TrajectoryStateOnSurface tsos =
	       tsoscomb.combine(meas.forwardPredictedState(),
				meas.backwardPredictedState());
	    if (tsos.isValid())
	    {
	       hitvec.push_back(hit);
	       tsosvec.push_back(tsos);
	    }
	 }
      }
    
      // transform RecHit vector to AlignableDet vector
      std::vector <AlignableDetOrUnitPtr> alidetvec = 
	 theAlignableDetAccessor->alignablesFromHits(hitvec);

      // get concatenated alignment parameters for list of alignables
      CompositeAlignmentParameters aap = 
	 theAlignmentParameterStore->selectParameters(alidetvec);

      std::vector<TrajectoryStateOnSurface>::const_iterator itsos=tsosvec.begin();
      std::vector<const TransientTrackingRecHit*>::const_iterator ihit=hitvec.begin();

      // loop over vectors(hit,tsos)
      while (itsos != tsosvec.end()) 
      {
	 // get AlignableDet for this hit
	 const GeomDet* det=(*ihit)->det();
	 AlignableDetOrUnitPtr alidet = 
	    theAlignableDetAccessor->alignableFromGeomDet(det);

	 // get relevant Alignable
	 Alignable* ali=aap.alignableFromAlignableDet(alidet);

	 if (ali!=0) {
	    // get trajectory impact point
	    LocalPoint alvec = (*itsos).localPosition();
	    AlgebraicVector pos(2);
	    pos[0]=alvec.x(); // local x
	    pos[1]=alvec.y(); // local y

	    // get impact point covariance
	    AlgebraicSymMatrix ipcovmat(2);
	    ipcovmat[0][0] = (*itsos).localError().positionError().xx();
	    ipcovmat[1][1] = (*itsos).localError().positionError().yy();
	    ipcovmat[0][1] = (*itsos).localError().positionError().xy();
   
	    // get hit local position and covariance
	    AlgebraicVector coor(2);
	    coor[0] = (*ihit)->localPosition().x();
	    coor[1] = (*ihit)->localPosition().y();

	    AlgebraicSymMatrix covmat(2);
	    covmat[0][0] = (*ihit)->localPositionError().xx();
	    covmat[1][1] = (*ihit)->localPositionError().yy();
	    covmat[0][1] = (*ihit)->localPositionError().xy();

	    // add hit and impact point covariance matrices
	    covmat = covmat + ipcovmat;

	    // calculate the x pull and y pull of this hit
	    double xpull = 0.;
	    double ypull = 0.;
	    if (covmat[0][0] != 0.) xpull = (pos[0] - coor[0])/sqrt(fabs(covmat[0][0]));
	    if (covmat[1][1] != 0.) ypull = (pos[1] - coor[1])/sqrt(fabs(covmat[1][1]));

	    // get Alignment Parameters
	    AlignmentParameters* params = ali->alignmentParameters();
	    // get derivatives
	    AlgebraicMatrix derivs=params->selectedDerivatives(*itsos,alidet);

	    // invert covariance matrix
	    int ierr; 
	    covmat.invert(ierr);
	    if (ierr != 0) { 
	       edm::LogError("AlignmentMonitorHIPNtuple") << "Matrix inversion failed!"; 
	       return; 
	    }

	    bool useThisHit = (theMaxAllowedHitPull <= 0.);

	    // ignore track minus center-of-chamber "residual" from 1d hits (only muon drift tubes)
	    if ((*ihit)->dimension() == 1) {
	       covmat[1][1] = 0.;
	       covmat[0][1] = 0.;

	       useThisHit = useThisHit || (fabs(xpull) < theMaxAllowedHitPull);
	    }
	    else {
	       useThisHit = useThisHit || (fabs(xpull) < theMaxAllowedHitPull  &&  fabs(ypull) < theMaxAllowedHitPull);
	    }

	    if (useThisHit) {
	       // calculate user parameters
	       int npar=derivs.num_row();
	       AlgebraicSymMatrix thisjtvj(npar);
	       AlgebraicVector thisjtve(npar);
	       thisjtvj=covmat.similarity(derivs);
	       thisjtve=derivs * covmat * (pos-coor);

	       m_ntuple_hitx = coor[0];
	       m_ntuple_hity = coor[1];
	       m_ntuple_resx = (pos[0] - coor[0]);
	       m_ntuple_resy = (pos[1] - coor[1]);

	       assert( npar == m_npar );  // yeah, that should be a ConfigError exception

// 	       int ierr; 
// 	       thisjtvj.invert(ierr);
// 	       if (ierr != 0) { 
// 		  edm::LogError("AlignmentMonitorHIPNtuple") << "thisjtvj matrix inversion failed!"; 
// 		  return; 
// 	       }

// 	       m_ntuple_rawid = ali->id();
// 	       m_ntuple_p1 = m_ntuple_p2 = m_ntuple_p3 = m_ntuple_p4 = m_ntuple_p5 = m_ntuple_p6 = 0.;
// 	       m_ntuple_e1 = m_ntuple_e2 = m_ntuple_e3 = m_ntuple_e4 = m_ntuple_e5 = m_ntuple_e6 = 0.;
// 	       m_ntuple_c12 = m_ntuple_c13 = m_ntuple_c14 = m_ntuple_c15 = m_ntuple_c16 = 0.;
// 	       m_ntuple_c23 = m_ntuple_c24 = m_ntuple_c25 = m_ntuple_c26 = 0.;
// 	       m_ntuple_c34 = m_ntuple_c35 = m_ntuple_c36 = 0.;
// 	       m_ntuple_c45 = m_ntuple_c46 = 0.;
// 	       m_ntuple_c56 = 0.;

// 	       if (m_npar >= 1) {
// 		  if (thisjtvj[1-1][1-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p1 error squared is " << thisjtvj[1-1][1-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p1 = thisjtve[1-1] * thisjtvj[1-1][1-1];
// 		  m_ntuple_e1 = sqrt(thisjtvj[1-1][1-1]);
// 	       }

// 	       if (m_npar >= 2) {
// 		  if (thisjtvj[2-1][2-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p2 error squared is " << thisjtvj[2-1][2-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p2 = thisjtve[2-1] * thisjtvj[2-1][2-1];
// 		  m_ntuple_e2 = sqrt(thisjtvj[2-1][2-1]);
// 		  m_ntuple_c12 = thisjtvj[1-1][2-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[2-1][2-1]);
// 	       }

// 	       if (m_npar >= 3) {
// 		  if (thisjtvj[3-1][3-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p3 error squared is " << thisjtvj[3-1][3-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p3 = thisjtve[3-1] * thisjtvj[3-1][3-1];
// 		  m_ntuple_e3 = sqrt(thisjtvj[3-1][3-1]);
// 		  m_ntuple_c13 = thisjtvj[1-1][3-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[3-1][3-1]);
// 		  m_ntuple_c23 = thisjtvj[2-1][3-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[3-1][3-1]);
// 	       }

// 	       if (m_npar >= 4) {
// 		  if (thisjtvj[4-1][4-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p4 error squared is " << thisjtvj[4-1][4-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p4 = thisjtve[4-1] * thisjtvj[4-1][4-1];
// 		  m_ntuple_e4 = sqrt(thisjtvj[4-1][4-1]);
// 		  m_ntuple_c14 = thisjtvj[1-1][4-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[4-1][4-1]);
// 		  m_ntuple_c24 = thisjtvj[2-1][4-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[4-1][4-1]);
// 		  m_ntuple_c34 = thisjtvj[3-1][4-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[4-1][4-1]);
// 	       }

// 	       if (m_npar >= 5) {
// 		  if (thisjtvj[5-1][5-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p5 error squared is " << thisjtvj[5-1][5-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p5 = thisjtve[5-1] * thisjtvj[5-1][5-1];
// 		  m_ntuple_e5 = sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c15 = thisjtvj[1-1][5-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c25 = thisjtvj[2-1][5-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c35 = thisjtvj[3-1][5-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c45 = thisjtvj[4-1][5-1] / sqrt(thisjtvj[4-1][4-1]) / sqrt(thisjtvj[5-1][5-1]);
// 	       }

// 	       if (m_npar >= 6) {
// 		  if (thisjtvj[6-1][6-1] <= 0.) {
// 		     edm::LogError("AlignmentMonitorHIPNtuple") << "p6 error squared is " << thisjtvj[6-1][6-1] << std::endl;
// 		     return;
// 		  }

// 		  m_ntuple_p6 = thisjtve[6-1] * thisjtvj[6-1][6-1];
// 		  m_ntuple_e6 = sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c16 = thisjtvj[1-1][6-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c26 = thisjtvj[2-1][6-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c36 = thisjtvj[3-1][6-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c46 = thisjtvj[4-1][6-1] / sqrt(thisjtvj[4-1][4-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c56 = thisjtvj[5-1][6-1] / sqrt(thisjtvj[5-1][5-1]) / sqrt(thisjtvj[6-1][6-1]);
// 	       }

	       m_ntuple_rawid = ali->id();
	       m_ntuple_time = iEvent.time().value();
	       m_ntuple_p1 = m_ntuple_p2 = m_ntuple_p3 = m_ntuple_p4 = m_ntuple_p5 = m_ntuple_p6 = 0.;
	       m_ntuple_e1 = m_ntuple_e2 = m_ntuple_e3 = m_ntuple_e4 = m_ntuple_e5 = m_ntuple_e6 = 0.;
// 	       m_ntuple_c12 = m_ntuple_c13 = m_ntuple_c14 = m_ntuple_c15 = m_ntuple_c16 = 0.;
// 	       m_ntuple_c23 = m_ntuple_c24 = m_ntuple_c25 = m_ntuple_c26 = 0.;
// 	       m_ntuple_c34 = m_ntuple_c35 = m_ntuple_c36 = 0.;
// 	       m_ntuple_c45 = m_ntuple_c46 = 0.;
// 	       m_ntuple_c56 = 0.;

	       DetId id = (*ihit)->geographicalId();

	       m_ntuple_ring = 0;
	       m_ntuple_chamber = 0;
	       if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
		  CSCDetId cscid(id.rawId());
		  m_ntuple_ring = cscid.ring();
		  m_ntuple_chamber = cscid.chamber();
	       }

	       m_ntuple_tanTheta = itsos->localMomentum().y() / itsos->localMomentum().z();

	       if (m_npar >= 1) {
		  if (1./thisjtvj[1-1][1-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p1 error squared is " << 1./thisjtvj[1-1][1-1] << std::endl;
		     return;
		  }

		  m_ntuple_p1 = thisjtve[1-1] / thisjtvj[1-1][1-1];
		  m_ntuple_e1 = sqrt(1./thisjtvj[1-1][1-1]);
	       }

	       if (m_npar >= 2) {
		  if (1./thisjtvj[2-1][2-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p2 error squared is " << 1./thisjtvj[2-1][2-1] << std::endl;
		     return;
		  }

		  m_ntuple_p2 = thisjtve[2-1] / thisjtvj[2-1][2-1];
		  m_ntuple_e2 = sqrt(1./thisjtvj[2-1][2-1]);
//		  m_ntuple_c12 = thisjtvj[1-1][2-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[2-1][2-1]);
	       }

	       if (m_npar >= 3) {
		  if (1./thisjtvj[3-1][3-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p3 error squared is " << 1./thisjtvj[3-1][3-1] << std::endl;
		     return;
		  }

		  m_ntuple_p3 = thisjtve[3-1] / thisjtvj[3-1][3-1];
		  m_ntuple_e3 = sqrt(1./thisjtvj[3-1][3-1]);
// 		  m_ntuple_c13 = thisjtvj[1-1][3-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[3-1][3-1]);
// 		  m_ntuple_c23 = thisjtvj[2-1][3-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[3-1][3-1]);
	       }

	       if (m_npar >= 4) {
		  if (1./thisjtvj[4-1][4-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p4 error squared is " << 1./thisjtvj[4-1][4-1] << std::endl;
		     return;
		  }

		  m_ntuple_p4 = thisjtve[4-1] / thisjtvj[4-1][4-1];
		  m_ntuple_e4 = sqrt(1./thisjtvj[4-1][4-1]);
// 		  m_ntuple_c14 = thisjtvj[1-1][4-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[4-1][4-1]);
// 		  m_ntuple_c24 = thisjtvj[2-1][4-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[4-1][4-1]);
// 		  m_ntuple_c34 = thisjtvj[3-1][4-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[4-1][4-1]);
	       }

	       if (m_npar >= 5) {
		  if (1./thisjtvj[5-1][5-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p5 error squared is " << 1./thisjtvj[5-1][5-1] << std::endl;
		     return;
		  }

		  m_ntuple_p5 = thisjtve[5-1] / thisjtvj[5-1][5-1];
		  m_ntuple_e5 = sqrt(1./thisjtvj[5-1][5-1]);
// 		  m_ntuple_c15 = thisjtvj[1-1][5-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c25 = thisjtvj[2-1][5-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c35 = thisjtvj[3-1][5-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[5-1][5-1]);
// 		  m_ntuple_c45 = thisjtvj[4-1][5-1] / sqrt(thisjtvj[4-1][4-1]) / sqrt(thisjtvj[5-1][5-1]);
	       }

	       if (m_npar >= 6) {
		  if (1./thisjtvj[6-1][6-1] <= 0.) {
		     edm::LogError("AlignmentMonitorHIPNtuple") << "p6 error squared is " << 1./thisjtvj[6-1][6-1] << std::endl;
		     return;
		  }

		  m_ntuple_p6 = thisjtve[6-1] / thisjtvj[6-1][6-1];
		  m_ntuple_e6 = sqrt(1./thisjtvj[6-1][6-1]);
// 		  m_ntuple_c16 = thisjtvj[1-1][6-1] / sqrt(thisjtvj[1-1][1-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c26 = thisjtvj[2-1][6-1] / sqrt(thisjtvj[2-1][2-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c36 = thisjtvj[3-1][6-1] / sqrt(thisjtvj[3-1][3-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c46 = thisjtvj[4-1][6-1] / sqrt(thisjtvj[4-1][4-1]) / sqrt(thisjtvj[6-1][6-1]);
// 		  m_ntuple_c56 = thisjtvj[5-1][6-1] / sqrt(thisjtvj[5-1][5-1]) / sqrt(thisjtvj[6-1][6-1]);
	       }

	       m_ntuple->Fill();
	    }
	 }

	 itsos++;
	 ihit++;
      } 
   } // end loop over track-trajectories
}

void AlignmentMonitorHIPNtuple::afterAlignment(const edm::EventSetup &iSetup) {}

//
// constructors and destructor
//

// AlignmentMonitorHIPNtuple::AlignmentMonitorHIPNtuple(const AlignmentMonitorHIPNtuple& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorHIPNtuple& AlignmentMonitorHIPNtuple::operator=(const AlignmentMonitorHIPNtuple& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorHIPNtuple temp(rhs);
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

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorHIPNtuple, "AlignmentMonitorHIPNtuple");
