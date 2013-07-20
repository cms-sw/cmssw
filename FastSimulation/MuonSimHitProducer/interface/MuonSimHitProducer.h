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
// $Id: MuonSimHitProducer.h,v 1.11 2013/02/27 22:22:53 wdd Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

// FastSimulation headers
class RandomEngine;
class MagneticField;
class TrackerGeometry;
class DTGeometry;
class CSCGeometry;
class RPCGeometry;
class MuonServiceProxy;
class MaterialEffects;
class TrajectoryStateOnSurface;
class Propagator;

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
      Chi2MeasurementEstimator theEstimator;

      const MagneticField*  magfield;
      const DTGeometry*     dtGeom;
      const CSCGeometry*    cscGeom;
      const RPCGeometry*    rpcGeom;
      const Propagator*     propagatorWithMaterial;
            Propagator* propagatorWithoutMaterial;

      MaterialEffects* theMaterialEffects;
  
      virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      void readParameters(const edm::ParameterSet&, 
			  const edm::ParameterSet&,
			  const edm::ParameterSet& );

      // Parameters to emulate the muonSimHit association inefficiency due to delta's
      double kDT;
      double fDT;
      double kCSC;
      double fCSC;

      /// Simulate material effects in iron (dE/dx, multiple scattering)
      void applyMaterialEffects(TrajectoryStateOnSurface& tsosWithdEdx,
				TrajectoryStateOnSurface& tsos,
				double radPath);

          
  // ----------- parameters ---------------------------- 
      bool fullPattern_;
      bool doL1_ , doL3_ , doGL_;
      std::string theSimModuleLabel_ , theSimModuleProcess_, theTrkModuleLabel_ ;
};

#endif
