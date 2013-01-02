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
// $Id: SiStripElectronAlgo.h,v 1.17 2011/03/21 17:10:32 innocent Exp $
//

// system include files

#include <map>

// user include files

// forward declarations

#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class TrackerTopology;

class SiStripElectronAlgo
{

   public:
      SiStripElectronAlgo(unsigned int maxHitsOnDetId,
			  double originUncertainty,
			  double phiBandWidth,
			  double maxNormResid,
			  unsigned int minHits,
			  double maxReducedChi2);

      virtual ~SiStripElectronAlgo();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      void prepareEvent(const edm::ESHandle<TrackerGeometry>& tracker,
			const edm::Handle<SiStripRecHit2DCollection>& rphiHits,
			const edm::Handle<SiStripRecHit2DCollection>& stereoHits,
			const edm::Handle<SiStripMatchedRecHit2DCollection>& matchedHits,
			const edm::ESHandle<MagneticField>& magneticField);

      // returns true iff an electron/positron was found
      // and inserts SiStripElectron and trackCandidate into electronOut and trackCandidateOut
      bool findElectron(reco::SiStripElectronCollection& electronOut,
			TrackCandidateCollection& trackCandidateOut,
			const reco::SuperClusterRef& superclusterIn,
			const TrackerTopology *tTopo);

   private:
      SiStripElectronAlgo(const SiStripElectronAlgo&); // stop default

      const SiStripElectronAlgo& operator=(const SiStripElectronAlgo&); // stop default

      // inserts pointers to good hits into hitPointersOut
      // selects hits on DetIds that have no more than maxHitsOnDetId_
      // selects from stereo if stereo == true, rphi otherwise
      // selects from TID or TEC if endcap == true, TIB or TOB otherwise
      void coarseHitSelection(std::vector<const SiStripRecHit2D*>& hitPointersOut,
			      const TrackerTopology *tTopo,
			      bool stereo, bool endcap);
      void coarseBarrelMonoHitSelection(std::vector<const SiStripRecHit2D*>& monoHitPointersOut );
      void coarseEndcapMonoHitSelection(std::vector<const SiStripRecHit2D*>& monoHitPointersOut );
      void coarseMatchedHitSelection(std::vector<const SiStripMatchedRecHit2D*>& coarseMatchedHitPointersOut);


      // projects a phi band of width phiBandWidth_ from supercluster into tracker (given a chargeHypothesis)
      // fills *_pos_ or *_neg_ member data with the results
      // returns true iff the electron/positron passes cuts
      bool projectPhiBand(float chargeHypothesis, 
			  const reco::SuperClusterRef& superclusterIn,
			  const TrackerTopology *tTopo);

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
      double maxNormResid_;
      unsigned int minHits_;
      double maxReducedChi2_;

      // changes with each event
      const TrackerGeometry* tracker_p_;
      const SiStripRecHit2DCollection* rphiHits_p_;
      const SiStripRecHit2DCollection* stereoHits_p_;
      const SiStripMatchedRecHit2DCollection* matchedHits_p_;
      const MagneticField* magneticField_p_;

      const edm::Handle<SiStripRecHit2DCollection>* rphiHits_hp_;
      const edm::Handle<SiStripRecHit2DCollection>* stereoHits_hp_;
      const edm::Handle<SiStripMatchedRecHit2DCollection>* matchedHits_hp_;


      std::map<const SiStripRecHit2D*, unsigned int> rphiKey_;
      std::map<const SiStripRecHit2D*, unsigned int> stereoKey_;
      std::map<const SiStripMatchedRecHit2D*, unsigned int> matchedKey_;

      std::map<const TrackingRecHit*, bool> hitUsed_;
      std::map<const TrackingRecHit*, bool> matchedHitUsed_;


      double redchi2_pos_;
      GlobalPoint position_pos_;
      GlobalVector momentum_pos_;
      const SiStripRecHit2D* innerhit_pos_;
      std::vector<const TrackingRecHit*> outputHits_pos_;
      std::vector<SiStripRecHit2D> outputRphiHits_pos_;
      std::vector<SiStripRecHit2D> outputStereoHits_pos_;
      std::vector<SiStripRecHit2D> outputMatchedHits_neg_;

      double phiVsRSlope_pos_;
      double slope_pos_;
      double intercept_pos_;
      double chi2_pos_;
      int ndof_pos_;
      double correct_pT_pos_;
      double pZ_pos_;
      double zVsRSlope_pos_;
      unsigned int numberOfMatchedHits_pos_;
      unsigned int numberOfStereoHits_pos_;
      unsigned int numberOfBarrelRphiHits_pos_;
      unsigned int numberOfEndcapZphiHits_pos_;
      
      double redchi2_neg_;
      GlobalPoint position_neg_;
      GlobalVector momentum_neg_;
      const SiStripRecHit2D* innerhit_neg_;
      std::vector<const TrackingRecHit*> outputHits_neg_;
      std::vector<SiStripRecHit2D> outputRphiHits_neg_;
      std::vector<SiStripRecHit2D> outputStereoHits_neg_;
      double phiVsRSlope_neg_;
      double slope_neg_;
      double intercept_neg_;
      double chi2_neg_;
      int ndof_neg_;
      double correct_pT_neg_;
      double pZ_neg_;
      double zVsRSlope_neg_;
      unsigned numberOfStereoHits_neg_;
      unsigned numberOfBarrelRphiHits_neg_;
      unsigned numberOfEndcapZphiHits_neg_;
};


#endif
