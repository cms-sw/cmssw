#ifndef EgammaElectronAlgos_SiStripElectronAlgo_h
#define EgammaElectronAlgos_SiStripElectronAlgo_h
// -*- C++ -*-
//
// Package:     EgammaElectronAlgos
// Class  :     SiStripElectronAlgo
// 
/**\class SiStripElectronAlgo SiStripElectronAlgo.h RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:11:58 EDT 2006
// $Id: SiStripElectronAlgo.h,v 1.4 2006/06/21 22:47:25 pivarski Exp $
//

// system include files

// user include files

// forward declarations

#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class SiStripElectronAlgo
{

   public:
      SiStripElectronAlgo(unsigned int maxHitsOnDetId,
			  double originUncertainty,
			  double phiBandWidth,
			  unsigned int minHits,
			  double maxReducedChi2);

      virtual ~SiStripElectronAlgo();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      void prepareEvent(const TrackerGeometry* tracker,
			const SiStripRecHit2DCollection* rphiHits,
			const SiStripRecHit2DCollection* stereoHits,
			const MagneticField* magneticField);

      // returns number of electrons found (0, 1, or 2),
      // inserts electrons and trajectories into electronOut and trajectoryOut
      int findElectron(reco::SiStripElectronCollection& electronOut,
		       TrackCandidateCollection& trackCandidateOut,
		       const reco::SuperClusterRef& superclusterIn);

   private:
      SiStripElectronAlgo(const SiStripElectronAlgo&); // stop default

      const SiStripElectronAlgo& operator=(const SiStripElectronAlgo&); // stop default

      // inserts pointers to good hits into hitPointersOut
      // selects hits on DetIds that have no more than maxHitsOnDetId_
      // selects from stereo if stereo == true, rphi otherwise
      // selects from TID or TEC if endcap == true, TIB or TOB otherwise
      void coarseHitSelection(std::vector<const SiStripRecHit2D*>& hitPointersOut,
			      bool stereo, bool endcap);

     // projects a phi band of width phiBandWidth_ from supercluster into tracker (given a chargeHypothesis)
     // copies and inserts passing hits into a TrackCandidate, which it puts into trackCandidateOut if passes cuts
     // returns true iff the electron/positron passes cuts
      bool projectPhiBand(reco::SiStripElectronCollection& electronOut,
			  TrackCandidateCollection& trackCandidateOut,
			  float chargeHypothesis,
			  const reco::SuperClusterRef& superclusterIn);

      double unwrapPhi(double phi) const {
	 while (phi > M_PI) { phi -= 2.*M_PI; }
	 while (phi < -M_PI) { phi += 2.*M_PI; }
	 return phi;
      }

      // ---------- member data --------------------------------

      // parameters
      unsigned int maxHitsOnDetId_;
      double originUncertainty_;
      double phiBandWidth_;
      unsigned int minHits_;
      double maxReducedChi2_;

      // changes with each event
      const TrackerGeometry* tracker_p;
      const SiStripRecHit2DCollection* rphiHits_p;
      const SiStripRecHit2DCollection* stereoHits_p;
      const MagneticField* magneticField_p;
};


#endif
