#ifndef ElectronSeedMerger_H
#define ElectronSeedMerger_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class ElectronSeedMerger : public edm::EDProducer {
   public:
      explicit ElectronSeedMerger(const edm::ParameterSet&);
      ~ElectronSeedMerger();
  
   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
 

      edm::ParameterSet conf_;

      ///SEED COLLECTIONS
      edm::InputTag ecalBasedSeeds_;
      edm::InputTag tkBasedSeeds_;


};
#endif
