#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"
#include <iostream>


namespace {
  struct Dumper {
    void visit(BlockWipedAllocator const& alloc) const {
      BlockWipedAllocator::Stat sa1 = alloc.stat();
      std::cout << "Alloc for size " << sa1.typeSize
		<< ": " << sa1.blockSize
		<< " " << sa1.currentOccupancy
		<< " " << sa1.currentAvailable
		<< " " << sa1.totalAvailable
		<< " " << sa1.nBlocks 
		<< std::endl;
    }
    
  };
  
}

/**  manage the allocator
 */
class BlockWipedAllocatorService {
public:
  BlockWipedAllocatorService(const edm::ParameterSet & iConfig,
			     edm::ActivityRegistry & iAR ) {
    
    iAR.watchPreProcessEvent(this,&BlockWipedAllocatorService::preEventProcessing);
    iAR.watchPostEndJob(this,&BlockWipedAllocatorService::postEndJob);
    iAR.watchPreModule(this,&BlockWipedAllocatorService::preModule);
    iAR.watchPostModule(this,&BlockWipedAllocatorService::postModule);
  }

  // wipe the workspace before each event
  void preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime) {
    std::cout << "BlockAllocator stat"<< std::endl;
    Dumper dumper;
    blockWipedPool().visit(dumper);
    blockWipedPool().wipe();
  }
 
  // wipe before each module
  void preModule(const edm::ModuleDescription& desc){
//     blockWipedPool().wipe();
  }

void postModule(const edm::ModuleDescription& desc){
//    std::cout << "BlockAllocator stat"<< std::endl;
//    Dumper dumper;
//    blockWipedPool().visit(dumper);
 }

  // final stat
  void postEndJob() {
    Dumper dumper;
    blockWipedPool().visit(dumper);
  }
  
};


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(BlockWipedAllocatorService);
