#ifndef FastSimulation_MuonSimHitProducer_MuonSimHitProducer_h
#define FastSimulation_MuonSimHitProducer_MuonSimHitProducer_h

//
// Package:    MuonSimHitProducer
// Class:      MuonSimHitProducer
// 
/**\class MuonSimHitProducer MuonSimHitProducer.cc FastSimulation/MuonSimHitProducer/src/MuonSimHitProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//  Author:  Martijn Mulders
// Created:  Wed July 11 12:37:24 CET 2007
// $Id: MuonSimHitProducer.h,v 1.3 2007/11/15 17:24:24 pjanot Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"

// FastSimulation headers
class MuonTrajectoryUpdator;
class RandomEngine;
class MagneticField;
class TrackerGeometry;
class DTGeometry;
class CSCGeometry;
class RPCGeometry;
class MuonServiceProxy;
class MaterialEffects;
class TrajectoryStateOnSurface;

/*
namespace reco { 
  class Muon;
}
*/

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

//
// class declaration
//

class MuonSimHitProducer : public edm::EDProducer {
   public:

      explicit MuonSimHitProducer(const edm::ParameterSet&);
      ~MuonSimHitProducer();

   private:

      const RandomEngine * random;
      MuonServiceProxy *theService;
      MuonTrajectoryUpdator *theUpdator;

      const MagneticField*  magfield;
      const DTGeometry*     dtGeom;
      const CSCGeometry*    cscGeom;
      const RPCGeometry*    rpcGeom;

      MaterialEffects* theMaterialEffects;
  
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void readParameters(const edm::ParameterSet&, 
			  const edm::ParameterSet&,
			  const edm::ParameterSet& );

      void applyScattering(TrajectoryStateOnSurface& tsos,
			   double pathLength);

          
  // ----------- parameters ---------------------------- 
      bool debug_;
      bool fullPattern_;
      bool doL1_ , doL3_ , doGL_;
      std::string theSimModuleLabel_ , theSimModuleProcess_, theTrkModuleLabel_ ;
      double minEta_ ,  maxEta_;
};

#endif
