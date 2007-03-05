/*
 * Seeds for Tracking of Laser Beams
 */

#ifndef LaserAlignment_LaserSeedGenerator_h
#define LaserAlignment_LaserSeedGenerator_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/LaserAlignment/interface/SeedGeneratorForLaserBeams.h"

//
// class decleration
//

class LaserSeedGenerator : public edm::EDProducer {
   public:
      explicit LaserSeedGenerator(const edm::ParameterSet&);
      ~LaserSeedGenerator();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      SeedGeneratorForLaserBeams laser_seed;
};

#endif
