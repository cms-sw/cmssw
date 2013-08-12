// -*- C++ -*-
//
// Package:     Services
// Class  :     Tracer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 14:17:58 EDT 2005
//

// user include files
#include "FWCore/Services/src/Tracer.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

// system include files
#include <iostream>

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Tracer::Tracer(ParameterSet const& iPS, ActivityRegistry&iRegistry) :
  indention_(iPS.getUntrackedParameter<std::string>("indention")),
  depth_(0) {
   iRegistry.watchPostBeginJob(this, &Tracer::postBeginJob);
   iRegistry.watchPostEndJob(this, &Tracer::postEndJob);

   iRegistry.watchPreSource(this, &Tracer::preSource);
   iRegistry.watchPostSource(this, &Tracer::postSource);

   iRegistry.watchPreSourceLumi(this, &Tracer::preSourceLumi);
   iRegistry.watchPostSourceLumi(this, &Tracer::postSourceLumi);

   iRegistry.watchPreSourceRun(this, &Tracer::preSourceRun);
   iRegistry.watchPostSourceRun(this, &Tracer::postSourceRun);

   iRegistry.watchPreOpenFile(this, &Tracer::preOpenFile);
   iRegistry.watchPostOpenFile(this, &Tracer::postOpenFile);

   iRegistry.watchPreCloseFile(this, &Tracer::preCloseFile);
   iRegistry.watchPostCloseFile(this, &Tracer::postCloseFile);

   iRegistry.watchPreModuleBeginStream(this, &Tracer::preModuleBeginStream);
   iRegistry.watchPostModuleBeginStream(this, &Tracer::postModuleBeginStream);

   iRegistry.watchPreModuleEndStream(this, &Tracer::preModuleEndStream);
   iRegistry.watchPostModuleEndStream(this, &Tracer::postModuleEndStream);

   iRegistry.watchPreGlobalBeginRun(this, &Tracer::preGlobalBeginRun);
   iRegistry.watchPostGlobalBeginRun(this, &Tracer::postGlobalBeginRun);

   iRegistry.watchPreGlobalEndRun(this, &Tracer::preGlobalEndRun);
   iRegistry.watchPostGlobalEndRun(this, &Tracer::postGlobalEndRun);

   iRegistry.watchPreStreamBeginRun(this, &Tracer::preStreamBeginRun);
   iRegistry.watchPostStreamBeginRun(this, &Tracer::postStreamBeginRun);

   iRegistry.watchPreStreamEndRun(this, &Tracer::preStreamEndRun);
   iRegistry.watchPostStreamEndRun(this, &Tracer::postStreamEndRun);

   iRegistry.watchPreGlobalBeginLumi(this, &Tracer::preGlobalBeginLumi);
   iRegistry.watchPostGlobalBeginLumi(this, &Tracer::postGlobalBeginLumi);

   iRegistry.watchPreGlobalEndLumi(this, &Tracer::preGlobalEndLumi);
   iRegistry.watchPostGlobalEndLumi(this, &Tracer::postGlobalEndLumi);

   iRegistry.watchPreStreamBeginLumi(this, &Tracer::preStreamBeginLumi);
   iRegistry.watchPostStreamBeginLumi(this, &Tracer::postStreamBeginLumi);

   iRegistry.watchPreStreamEndLumi(this, &Tracer::preStreamEndLumi);
   iRegistry.watchPostStreamEndLumi(this, &Tracer::postStreamEndLumi);

   iRegistry.watchPreEvent(this, &Tracer::preEvent);
   iRegistry.watchPostEvent(this, &Tracer::postEvent);

   iRegistry.watchPrePathEvent(this, &Tracer::prePathEvent);
   iRegistry.watchPostPathEvent(this, &Tracer::postPathEvent);

   iRegistry.watchPreModuleConstruction(this, &Tracer::preModuleConstruction);
   iRegistry.watchPostModuleConstruction(this, &Tracer::postModuleConstruction);

   iRegistry.watchPreModuleBeginJob(this, &Tracer::preModuleBeginJob);
   iRegistry.watchPostModuleBeginJob(this, &Tracer::postModuleBeginJob);

   iRegistry.watchPreModuleEndJob(this, &Tracer::preModuleEndJob);
   iRegistry.watchPostModuleEndJob(this, &Tracer::postModuleEndJob);

   iRegistry.watchPreModuleEvent(this, &Tracer::preModuleEvent);
   iRegistry.watchPostModuleEvent(this, &Tracer::postModuleEvent);

   iRegistry.watchPreModuleStreamBeginRun(this, &Tracer::preModuleStreamBeginRun);
   iRegistry.watchPostModuleStreamBeginRun(this, &Tracer::postModuleStreamBeginRun);
   iRegistry.watchPreModuleStreamEndRun(this, &Tracer::preModuleStreamEndRun);
   iRegistry.watchPostModuleStreamEndRun(this, &Tracer::postModuleStreamEndRun);

   iRegistry.watchPreModuleStreamBeginLumi(this, &Tracer::preModuleStreamBeginLumi);
   iRegistry.watchPostModuleStreamBeginLumi(this, &Tracer::postModuleStreamBeginLumi);
   iRegistry.watchPreModuleStreamEndLumi(this, &Tracer::preModuleStreamEndLumi);
   iRegistry.watchPostModuleStreamEndLumi(this, &Tracer::postModuleStreamEndLumi);

   iRegistry.watchPreModuleGlobalBeginRun(this, &Tracer::preModuleGlobalBeginRun);
   iRegistry.watchPostModuleGlobalBeginRun(this, &Tracer::postModuleGlobalBeginRun);
   iRegistry.watchPreModuleGlobalEndRun(this, &Tracer::preModuleGlobalEndRun);
   iRegistry.watchPostModuleGlobalEndRun(this, &Tracer::postModuleGlobalEndRun);

   iRegistry.watchPreModuleGlobalBeginLumi(this, &Tracer::preModuleGlobalBeginLumi);
   iRegistry.watchPostModuleGlobalBeginLumi(this, &Tracer::postModuleGlobalBeginLumi);
   iRegistry.watchPreModuleGlobalEndLumi(this, &Tracer::preModuleGlobalEndLumi);
   iRegistry.watchPostModuleGlobalEndLumi(this, &Tracer::postModuleGlobalEndLumi);

   iRegistry.watchPreSourceConstruction(this, &Tracer::preSourceConstruction);
   iRegistry.watchPostSourceConstruction(this, &Tracer::postSourceConstruction);
}

// Tracer::Tracer(Tracer const& rhs)
// {
//    // do actual copying here;
// }

//Tracer::~Tracer()
//{
//}

void
Tracer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

    edm::ParameterSetDescription desc;
    desc.addUntracked<std::string>("indention", "++")->setComment("Prefix characters for output. The characters are repeated to form the indentation.");
    descriptions.add("Tracer", desc);
   descriptions.setComment("This service prints each phase the framework is processing, e.g. constructing a module,running a module, etc.");
}

//
// assignment operators
//
// Tracer const& Tracer::operator=(Tracer const& rhs)
// {
//   //An exception safe implementation is
//   Tracer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
Tracer::postBeginJob() {
   std::cout << indention_ << " Job started" << std::endl;
}

void 
Tracer::postEndJob() {
   std::cout << indention_ << " Job ended" << std::endl;
}

void
Tracer::preSource() {
  std::cout << indention_ << indention_ << " source event" << std::endl;
}

void
Tracer::postSource() {
  std::cout << indention_ << indention_ << " finished: source event" << std::endl;
}

void
Tracer::preSourceLumi() {
  std::cout << indention_ << indention_ << " source lumi" << std::endl;
}

void
Tracer::postSourceLumi () {
  std::cout << indention_ << indention_ << " finished: source lumi" << std::endl;
}

void
Tracer::preSourceRun() {
  std::cout << indention_ << indention_ << " source run" << std::endl;
}

void
Tracer::postSourceRun () {
  std::cout << indention_ << indention_ << " finished: source run" << std::endl;
}

void
Tracer::preOpenFile(std::string const& lfn, bool) {
  std::cout << indention_ << indention_ << " open input file: " << lfn << std::endl;
}

void
Tracer::postOpenFile (std::string const&, bool) {
  std::cout << indention_ << indention_ << " finished: open input file" << std::endl;
}

void
Tracer::preCloseFile(std::string const & lfn, bool) {
  std::cout << indention_ << indention_ << " close input file: " << lfn << std::endl;
}
void
Tracer::postCloseFile (std::string const&, bool) {
  std::cout << indention_ << indention_ << " finished: close input file" << std::endl;
}

void
Tracer::preModuleBeginStream(StreamContext const&, ModuleDescription const& desc) {
  std::cout << indention_ << indention_ << " ModuleBeginStream: " << desc.moduleLabel() << std::endl; 
}

void
Tracer::postModuleBeginStream(StreamContext const&, ModuleDescription const&) {
  std::cout << indention_ << indention_ << " ModuleBeginStream finished" << std::endl; 
}

void
Tracer::preModuleEndStream(StreamContext const&, ModuleDescription const&) {
  std::cout << indention_ << indention_ << " ModuleEndStream: " << std::endl; 
}

void
Tracer::postModuleEndStream(StreamContext const&, ModuleDescription const&) {
  std::cout << indention_ << indention_ << " ModuleEndStream finished" << std::endl; 
}

void
Tracer::preGlobalBeginRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginRun run: " << gc.luminosityBlockID().run() 
            << " time: " << gc.timestamp().value() << std::endl; 
}

void
Tracer::postGlobalBeginRun(GlobalContext const&) {
  std::cout << indention_ << indention_ << " GlobalBeginRun finished" << std::endl; 
}

void
Tracer::preGlobalEndRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndRun run: " << gc.luminosityBlockID().run() 
            << " time: " << gc.timestamp().value() << std::endl; 
}

void
Tracer::postGlobalEndRun(GlobalContext const&, Run const&, EventSetup const&) {
  std::cout << indention_ << indention_ << " GlobalEndRun finished" << std::endl; 
}

void
Tracer::preStreamBeginRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginRun run: " << sc.eventID().run() 
            << " time: " << sc.timestamp().value() << std::endl; 
}

void
Tracer::postStreamBeginRun(StreamContext const&) {
  std::cout << indention_ << indention_ << " StreamBeginRun finished" << std::endl; 
}

void
Tracer::preStreamEndRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndRun run: " << sc.eventID().run() 
            << " time: " << sc.timestamp().value() << std::endl; 
}

void
Tracer::postStreamEndRun(StreamContext const&, Run const&, EventSetup const&) {
  std::cout << indention_ << indention_ << " StreamEndRun finished" << std::endl; 
}

void
Tracer::preGlobalBeginLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginLumi run: " << gc.luminosityBlockID().run() 
            << " lumi: " << gc.luminosityBlockID().luminosityBlock() << " time: " << gc.timestamp().value() << std::endl; 
}

void
Tracer::postGlobalBeginLumi(GlobalContext const&) {
  std::cout << indention_ << indention_ << " GlobalBeginLumi finished" << std::endl; 
}

void
Tracer::preGlobalEndLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndLumi run: " << gc.luminosityBlockID().run() 
            << " lumi: " << gc.luminosityBlockID().luminosityBlock() << " time: " << gc.timestamp().value() << std::endl; 
}

void
Tracer::postGlobalEndLumi(GlobalContext const&, LuminosityBlock const&, EventSetup const&) {
  std::cout << indention_ << indention_ << " GlobalEndLumi finished" << std::endl; 
}

void
Tracer::preStreamBeginLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginLumi run: " << sc.eventID().run() 
            << " lumi: " << sc.eventID().luminosityBlock() << " time: " << sc.timestamp().value() << std::endl; 
}

void
Tracer::postStreamBeginLumi(StreamContext const&) {
  std::cout << indention_ << indention_ << " StreamBeginLumi finished" << std::endl; 
}

void
Tracer::preStreamEndLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndLumi run: " << sc.eventID().run() 
            << " lumi: " << sc.eventID().luminosityBlock() << " time: " << sc.timestamp().value() << std::endl; 
}

void
Tracer::postStreamEndLumi(StreamContext const&, LuminosityBlock const&, EventSetup const&) {
  std::cout << indention_ << indention_ << " StreamEndLumi finished" << std::endl; 
}

void
Tracer::preEvent(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " processing event " << sc.eventID() << " time:" << sc.timestamp().value() << std::endl;
}

void
Tracer::postEvent(StreamContext const&, Event const&, EventSetup const&) {
  std::cout << indention_ << indention_ << " event finished" << std::endl;
}

void
Tracer::prePathEvent(StreamContext const&, PathContext const& pc) {
  std::cout << indention_ << indention_ << indention_ << " processing path for event: " << pc.pathName() << std::endl;
}

void
Tracer::postPathEvent(StreamContext const&, PathContext const& pc, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " path for event finished: " << pc.pathName() << std::endl;
}

void 
Tracer::preModuleConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " constructing module: " << desc.moduleLabel() << std::endl;
}

void 
Tracer::postModuleConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " construction finished: " << desc.moduleLabel() << std::endl;
}

void 
Tracer::preModuleBeginJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleBeginJob: " << desc.moduleLabel() << std::endl;
}

void 
Tracer::postModuleBeginJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleBeginJob finished: " << desc.moduleLabel() << std::endl;
}

void 
Tracer::preModuleEndJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " ModuleEndJob: " << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::postModuleEndJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " ModuleEndJob finished: " << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for event: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void 
Tracer::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " finished module for event: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}


void
Tracer::preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleStreamBeginRun: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleStreamBeginRun finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleStreamEndRun: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleStreamEndRun finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleStreamBeginLumi: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleStreamBeginLumi finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleStreamEndLumi: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleStreamEndLumi finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleGlobalBeginRun: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleGlobalBeginRun finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleGlobalEndRun: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleGlobalEndRun finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleGlobalBeginLumi: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleGlobalBeginLumi finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
   ++depth_;
   std::cout << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " ModuleGlobalEndLumi: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void
Tracer::postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
   --depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }   
   std::cout << " ModuleGlobalEndLumi finished: " << mcc.moduleDescription()->moduleLabel() << std::endl;
}

void 
Tracer::preSourceConstruction(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " constructing source:" << iDescription.moduleName() << std::endl;
}

void 
Tracer::postSourceConstruction(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " construction finished:" << iDescription.moduleName() << std::endl;
}
