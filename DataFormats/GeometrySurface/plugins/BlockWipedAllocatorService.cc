#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"
#include <iostream>


#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"


namespace {
  struct Dumper {
    void visit(BlockWipedAllocator const& alloc) const {
      BlockWipedAllocator::Stat sa1 = alloc.stat();
      std::cout << "Alloc for size " << sa1.typeSize
		<< ": " << sa1.blockSize
		<< ", " << sa1.currentOccupancy
		<< "/" << sa1.currentAvailable
		<< ", " << sa1.totalAvailable
		<< "/" << sa1.nBlocks
		<< ", " << sa1.alive
		<< std::endl;
    }
    
  };
  
}

/**  manage the allocator
 */
class BlockWipedAllocatorService {
private:
  bool m_useAlloc;
public:
  BlockWipedAllocatorService(const edm::ParameterSet & iConfig,
			     edm::ActivityRegistry & iAR ) {
    
    m_useAlloc = iConfig.getUntrackedParameter<bool>("usePoolAllocator",false);
    if (m_useAlloc) BlockWipedPoolAllocated::usePool();
    iAR.watchPreSource(this,&BlockWipedAllocatorService::preSource);
    iAR.watchPreProcessEvent(this,&BlockWipedAllocatorService::preEventProcessing);
    iAR.watchPostEndJob(this,&BlockWipedAllocatorService::postEndJob);
    iAR.watchPreModule(this,&BlockWipedAllocatorService::preModule);
    iAR.watchPostModule(this,&BlockWipedAllocatorService::postModule);
  }

  // wipe the workspace before each event
  void preEventProcessing(const edm::EventID&, const edm::Timestamp&) { wiper();}

  // nope event-principal deleted in source
  void preSource() {
   // wiper();
  }

  void dump() {
    std::cout << "ReferenceCounted stat"<< std::endl;
    std::cout << "still alive/referenced " 
	      << ReferenceCountedPoolAllocated::s_alive
	      << "/" << ReferenceCountedPoolAllocated::s_referenced
	      << std::endl;

    std::cout << "BlockAllocator stat"<< std::endl;
    std::cout << "still alive " << BlockWipedPoolAllocated::s_alive << std::endl;
    Dumper dumper;
    blockWipedPool().visit(dumper);
  }


  void wiper() {
    dump();
    blockWipedPool().wipe();
    blockWipedPool().clear();  // try to crash
    {
       static int c=0;
       if (20==c) {
       blockWipedPool().clear();
       c=0;
       }
       c++;
    }

  }
 
  // wipe before each module (no, obj in event....)
  void preModule(const edm::ModuleDescription& desc){
    //     blockWipedPool().wipe();
  }

  void postModule(const edm::ModuleDescription& desc){
    dump();
    }

  // final stat
  void postEndJob() {
    wiper();
  }
  
};



DEFINE_FWK_SERVICE(BlockWipedAllocatorService);
