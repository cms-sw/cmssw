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
// $Id: SiStripElectronAlgo.h,v 1.1 2006/05/27 04:31:25 pivarski Exp $
//

// system include files

// user include files

// forward declarations

#include "DataFormats/EgammaCandidates/interface/SiStripElectronCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

class SiStripElectronAlgo
{

   public:
      SiStripElectronAlgo(unsigned int maxHitsOnDetId,
			  double wedgePhiWidth,
			  double originUncertainty,
			  double deltaPhi,
			  unsigned int numHitsMin,
			  unsigned int numHitsMax);

      virtual ~SiStripElectronAlgo();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      void prepareEvent(const TrackerGeometry* tracker,
			const SiStripRecHit2DLocalPosCollection* rphiHits,
			const SiStripRecHit2DLocalPosCollection* stereoHits,
			const MagneticField* magneticField);

      bool bandHitCounting(reco::SiStripElectronCandidateCollection& electronOut,
			   const reco::SuperCluster& supercluster);

   private:
      SiStripElectronAlgo(const SiStripElectronAlgo&); // stop default

      const SiStripElectronAlgo& operator=(const SiStripElectronAlgo&); // stop default

      void getPoints(std::vector<GlobalPoint>& rphiBarrelPoints,
		     std::vector<GlobalPoint>& stereoBarrelPoints,
		     std::vector<GlobalPoint>& endcapPoints);
      double unwrapPhi(double phi) const {
	 while (phi > M_PI) { phi -= 2.*M_PI; }
	 while (phi < -M_PI) { phi += 2.*M_PI; }
	 return phi;
      }

      // ---------- member data --------------------------------

      // parameters
      unsigned int maxHitsOnDetId_;
      double wedgePhiWidth_;
      double originUncertainty_;
      double deltaPhi_;
      unsigned int numHitsMin_;
      unsigned int numHitsMax_;

      // changes with each event
      const TrackerGeometry* tracker_p;
      const SiStripRecHit2DLocalPosCollection* rphiHits_p;
      const SiStripRecHit2DLocalPosCollection* stereoHits_p;
      const MagneticField* magneticField_p;
};


#endif
