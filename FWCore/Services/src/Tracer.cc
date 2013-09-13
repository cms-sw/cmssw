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
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

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
#include "DataFormats/Common/interface/HLTPathStatus.h"
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
  depth_(0),
  dumpContextForLabel_(iPS.getUntrackedParameter<std::string>("dumpContextForLabel")),
  dumpNonModuleContext_(iPS.getUntrackedParameter<bool>("dumpNonModuleContext"))
 {
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
    desc.addUntracked<std::string>("dumpContextForLabel", "")->setComment("Prints context information to cout for the module transitions associated with this module label");
    desc.addUntracked<bool>("dumpNonModuleContext", false)->setComment("Prints context information to cout for the transitions not associated with any module label");
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
Tracer::preOpenFile(std::string const& lfn, bool b) {
  std::cout << indention_ << indention_ << " open input file: " << lfn;
  if(dumpNonModuleContext_) std::cout << " usedFallBack = " << b;
  std::cout << std::endl;
}

void
Tracer::postOpenFile (std::string const& lfn, bool b) {
  std::cout << indention_ << indention_ << " finished: open input file";
  if(dumpNonModuleContext_) std::cout << ": " << lfn << " usedFallBack = " << b;
  std::cout << std::endl;
}

void
Tracer::preCloseFile(std::string const & lfn, bool b) {
  std::cout << indention_ << indention_ << " close input file: " << lfn;
  if(dumpNonModuleContext_) std::cout << " usedFallBack = " << b;
  std::cout << std::endl;
}
void
Tracer::postCloseFile (std::string const& lfn, bool b) {
  std::cout << indention_ << indention_ << " finished: close input file";
  if(dumpNonModuleContext_) std::cout << ": " << lfn << " usedFallBack = " << b;
  std::cout << std::endl;
}

void
Tracer::preModuleBeginStream(StreamContext const& sc, ModuleDescription const& desc) {
  std::cout << indention_ << indention_ << " ModuleBeginStream: " << desc.moduleLabel(); 
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << sc;
    std::cout << desc;
  }
  std::cout << std::endl;
}

void
Tracer::postModuleBeginStream(StreamContext const& sc, ModuleDescription const& desc) {
  std::cout << indention_ << indention_ << " ModuleBeginStream finished"; 
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << sc;
    std::cout << desc;
  }
  std::cout << std::endl;
}

void
Tracer::preModuleEndStream(StreamContext const& sc, ModuleDescription const& desc) {
  std::cout << indention_ << indention_ << " ModuleEndStream: "; 
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << sc;
    std::cout << desc;
  }
  std::cout << std::endl;
}

void
Tracer::postModuleEndStream(StreamContext const& sc, ModuleDescription const& desc) {
  std::cout << indention_ << indention_ << " ModuleEndStream finished"; 
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << sc;
    std::cout << desc;
  }
  std::cout << std::endl;
}

void
Tracer::preGlobalBeginRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginRun run: " << gc.luminosityBlockID().run() 
            << " time: " << gc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::postGlobalBeginRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginRun finished\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::preGlobalEndRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndRun run: " << gc.luminosityBlockID().run() 
            << " time: " << gc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::postGlobalEndRun(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndRun finished\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::preStreamBeginRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginRun run: " << sc.eventID().run() 
            << " time: " << sc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::postStreamBeginRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginRun finished\n"; 
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::preStreamEndRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndRun run: " << sc.eventID().run() 
            << " time: " << sc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::postStreamEndRun(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndRun finished\n"; 
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::preGlobalBeginLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginLumi run: " << gc.luminosityBlockID().run() 
            << " lumi: " << gc.luminosityBlockID().luminosityBlock() << " time: " << gc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::postGlobalBeginLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalBeginLumi finished\n";  
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::preGlobalEndLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndLumi run: " << gc.luminosityBlockID().run() 
            << " lumi: " << gc.luminosityBlockID().luminosityBlock() << " time: " << gc.timestamp().value() << "\n"; 
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::postGlobalEndLumi(GlobalContext const& gc) {
  std::cout << indention_ << indention_ << " GlobalEndLumi finished\n";
  if(dumpNonModuleContext_) {
    std::cout << gc;
  }
  std::cout.flush();
}

void
Tracer::preStreamBeginLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginLumi run: " << sc.eventID().run() 
            << " lumi: " << sc.eventID().luminosityBlock() << " time: " << sc.timestamp().value() << "\n";  
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::postStreamBeginLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamBeginLumi finished\n";  
  if(dumpNonModuleContext_) {
    std::cout << "\n" << sc;
  }
  std::cout.flush();
}

void
Tracer::preStreamEndLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndLumi run: " << sc.eventID().run() 
            << " lumi: " << sc.eventID().luminosityBlock() << " time: " << sc.timestamp().value() << "\n";
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::postStreamEndLumi(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " StreamEndLumi finished\n"; 
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::preEvent(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " processing event " << sc.eventID() << " time:" << sc.timestamp().value() << "\n";
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::postEvent(StreamContext const& sc) {
  std::cout << indention_ << indention_ << " event finished\n";
  if(dumpNonModuleContext_) {
    std::cout << sc;
  }
  std::cout.flush();
}

void
Tracer::prePathEvent(StreamContext const& sc, PathContext const& pc) {
  std::cout << indention_ << indention_ << indention_ << " processing path for event: " << pc.pathName() << "\n";
  if(dumpNonModuleContext_) {
    std::cout << sc;
    std::cout << pc;
  }
  std::cout.flush();
}

void
Tracer::postPathEvent(StreamContext const& sc, PathContext const& pc, HLTPathStatus const& hlts) {
  std::cout << indention_ << indention_ << indention_ << " path for event finished: " << pc.pathName() << "\n";
  if(dumpNonModuleContext_) {
    std::cout << sc;
    std::cout << pc;
  }
  std::cout.flush();
}

void 
Tracer::preModuleConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " constructing module: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::postModuleConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " construction finished: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::preModuleBeginJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleBeginJob: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::postModuleBeginJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleBeginJob finished: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::preModuleEndJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleEndJob: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::postModuleEndJob(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " ModuleEndJob finished: " << desc.moduleLabel();
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;  
}

void 
Tracer::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " module for event: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void 
Tracer::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " finished module for event: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}


void
Tracer::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleStreamBeginRun: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleStreamBeginRun finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleStreamEndRun: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleStreamEndRun finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleStreamBeginLumi: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleStreamBeginLumi finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleStreamEndLumi: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleStreamEndLumi finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << sc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleGlobalBeginRun: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleGlobalBeginRun finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleGlobalEndRun: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleGlobalEndRun finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleGlobalBeginLumi: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleGlobalBeginLumi finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  ++depth_;
  std::cout << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }
  std::cout << " ModuleGlobalEndLumi: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void
Tracer::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  --depth_;
  std::cout << indention_ << indention_ << indention_;
  for(unsigned int depth = 0; depth !=depth_; ++depth) {
    std::cout << indention_;
  }   
  std::cout << " ModuleGlobalEndLumi finished: " << mcc.moduleDescription()->moduleLabel() << "\n";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    std::cout << gc;
    std::cout << mcc;
  }
  std::cout.flush();
}

void 
Tracer::preSourceConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " constructing source:" << desc.moduleName();
  if(dumpNonModuleContext_) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;
}

void 
Tracer::postSourceConstruction(ModuleDescription const& desc) {
  std::cout << indention_;
  std::cout << " construction finished:" << desc.moduleName();
  if(dumpNonModuleContext_) {
    std::cout << "\n" << desc;
  }
  std::cout << std::endl;
}
