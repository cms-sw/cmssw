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

   iRegistry.watchPreModule(this, &Tracer::preModuleEvent);
   iRegistry.watchPostModule(this, &Tracer::postModuleEvent);
   
   iRegistry.watchPreSourceConstruction(this, &Tracer::preSourceConstruction);
   iRegistry.watchPostSourceConstruction(this, &Tracer::postSourceConstruction);

   iRegistry.watchPreModuleConstruction(this, &Tracer::preModuleConstruction);
   iRegistry.watchPostModuleConstruction(this, &Tracer::postModuleConstruction);

   iRegistry.watchPreModuleBeginJob(this, &Tracer::preModuleBeginJob);
   iRegistry.watchPostModuleBeginJob(this, &Tracer::postModuleBeginJob);

   iRegistry.watchPreModuleEndJob(this, &Tracer::preModuleEndJob);
   iRegistry.watchPostModuleEndJob(this, &Tracer::postModuleEndJob);

   iRegistry.watchPreModuleBeginRun(this, &Tracer::preModuleBeginRun);
   iRegistry.watchPostModuleBeginRun(this, &Tracer::postModuleBeginRun);

   iRegistry.watchPreModuleEndRun(this, &Tracer::preModuleEndRun);
   iRegistry.watchPostModuleEndRun(this, &Tracer::postModuleEndRun);

   iRegistry.watchPreModuleBeginLumi(this, &Tracer::preModuleBeginLumi);
   iRegistry.watchPostModuleBeginLumi(this, &Tracer::postModuleBeginLumi);

   iRegistry.watchPreModuleEndLumi(this, &Tracer::preModuleEndLumi);
   iRegistry.watchPostModuleEndLumi(this, &Tracer::postModuleEndLumi);

   iRegistry.watchPreProcessPath(this, &Tracer::prePathEvent);
   iRegistry.watchPostProcessPath(this, &Tracer::postPathEvent);

   iRegistry.watchPrePathBeginRun(this, &Tracer::prePathBeginRun);
   iRegistry.watchPostPathBeginRun(this, &Tracer::postPathBeginRun);

   iRegistry.watchPrePathEndRun(this, &Tracer::prePathEndRun);
   iRegistry.watchPostPathEndRun(this, &Tracer::postPathEndRun);

   iRegistry.watchPrePathBeginLumi(this, &Tracer::prePathBeginLumi);
   iRegistry.watchPostPathBeginLumi(this, &Tracer::postPathBeginLumi);

   iRegistry.watchPrePathEndLumi(this, &Tracer::prePathEndLumi);
   iRegistry.watchPostPathEndLumi(this, &Tracer::postPathEndLumi);

   iRegistry.watchPreProcessEvent(this, &Tracer::preEvent);
   iRegistry.watchPostProcessEvent(this, &Tracer::postEvent);

   iRegistry.watchPreBeginRun(this, &Tracer::preBeginRun);
   iRegistry.watchPostBeginRun(this, &Tracer::postBeginRun);

   iRegistry.watchPreEndRun(this, &Tracer::preEndRun);
   iRegistry.watchPostEndRun(this, &Tracer::postEndRun);

   iRegistry.watchPreBeginLumi(this, &Tracer::preBeginLumi);
   iRegistry.watchPostBeginLumi(this, &Tracer::postBeginLumi);

   iRegistry.watchPreEndLumi(this, &Tracer::preEndLumi);
   iRegistry.watchPostEndLumi(this, &Tracer::postEndLumi);

   iRegistry.watchPreSource(this, &Tracer::preSourceEvent);
   iRegistry.watchPostSource(this, &Tracer::postSourceEvent);

   iRegistry.watchPreOpenFile(this, &Tracer::preOpenFile);
   iRegistry.watchPostOpenFile(this, &Tracer::postOpenFile);

   iRegistry.watchPreCloseFile(this, &Tracer::preCloseFile);
   iRegistry.watchPostCloseFile(this, &Tracer::postCloseFile);

   iRegistry.watchPreSourceRun(this, &Tracer::preSourceRun);
   iRegistry.watchPostSourceRun(this, &Tracer::postSourceRun);

   iRegistry.watchPreSourceLumi(this, &Tracer::preSourceLumi);
   iRegistry.watchPostSourceLumi(this, &Tracer::postSourceLumi);

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
Tracer::preSourceEvent() {
  std::cout << indention_ << indention_ << "source event" << std::endl;
}
void
Tracer::postSourceEvent () {
  std::cout << indention_ << indention_ << "finished: source event" << std::endl;
}

void
Tracer::preSourceLumi() {
  std::cout << indention_ << indention_ << "source lumi" << std::endl;
}
void
Tracer::postSourceLumi () {
  std::cout << indention_ << indention_ << "finished: source lumi" << std::endl;
}

void
Tracer::preSourceRun() {
  std::cout << indention_ << indention_ << "source run" << std::endl;
}
void
Tracer::postSourceRun () {
  std::cout << indention_ << indention_ << "finished: source run" << std::endl;
}

void
Tracer::preOpenFile() {
  std::cout << indention_ << indention_ << "open input file" << std::endl;
}
void
Tracer::postOpenFile () {
  std::cout << indention_ << indention_ << "finished: open input file" << std::endl;
}

void
Tracer::preCloseFile(std::string const & lfn, bool) {
  std::cout << indention_ << indention_ << "close input file " << lfn << std::endl;
}
void
Tracer::postCloseFile () {
  std::cout << indention_ << indention_ << "finished: close input file" << std::endl;
}

void 
Tracer::preEvent(EventID const& iID, Timestamp const& iTime) {
   depth_=0;
   std::cout << indention_ << indention_ << " processing event:" << iID << " time:" << iTime.value() << std::endl;
}
void 
Tracer::postEvent(Event const&, EventSetup const&) {
   std::cout << indention_ << indention_ << " finished event:" << std::endl;
}

void 
Tracer::prePathEvent(std::string const& iName) {
  std::cout << indention_ << indention_ << indention_ << " processing path for event:" << iName << std::endl;
}
void 
Tracer::postPathEvent(std::string const& iName, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " finished path for event:" << iName << std::endl;
}

void 
Tracer::preModuleEvent(ModuleDescription const& iDescription) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for event:" << iDescription.moduleLabel() << std::endl;
}
void 
Tracer::postModuleEvent(ModuleDescription const& iDescription) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   
   std::cout << " finished for event:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preBeginRun(RunID const& iID, Timestamp const& iTime) {
   depth_=0;
   std::cout << indention_ << indention_ << " processing begin run:" << iID << " time:" << iTime.value() << std::endl;
}
void 
Tracer::postBeginRun(Run const&, EventSetup const&) {
   std::cout << indention_ << indention_ << " finished begin run:" << std::endl;
}

void 
Tracer::prePathBeginRun(std::string const& iName) {
  std::cout << indention_ << indention_ << indention_ << " processing path for begin run:" << iName << std::endl;
}
void 
Tracer::postPathBeginRun(std::string const& iName, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " finished path for begin run:" << iName << std::endl;
}

void 
Tracer::preModuleBeginRun(ModuleDescription const& iDescription) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for begin run:" << iDescription.moduleLabel() << std::endl;
}
void 
Tracer::postModuleBeginRun(ModuleDescription const& iDescription) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   
   std::cout << " finished for begin run:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preEndRun(RunID const& iID, Timestamp const& iTime) {
   depth_=0;
   std::cout << indention_ << indention_ << " processing end run:" << iID << " time:" << iTime.value() << std::endl;
}
void 
Tracer::postEndRun(Run const&, EventSetup const&) {
   std::cout << indention_ << indention_ << " finished end run:" << std::endl;
}

void 
Tracer::prePathEndRun(std::string const& iName) {
  std::cout << indention_ << indention_ << indention_ << " processing path for end run:" << iName << std::endl;
}
void 
Tracer::postPathEndRun(std::string const& iName, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " finished path for end run:" << iName << std::endl;
}

void 
Tracer::preModuleEndRun(ModuleDescription const& iDescription) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for end run:" << iDescription.moduleLabel() << std::endl;
}
void 
Tracer::postModuleEndRun(ModuleDescription const& iDescription) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   
   std::cout << " finished for end run:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preBeginLumi(LuminosityBlockID const& iID, Timestamp const& iTime) {
   depth_=0;
   std::cout << indention_ << indention_ << " processing begin lumi:" << iID << " time:" << iTime.value() << std::endl;
}
void 
Tracer::postBeginLumi(LuminosityBlock const&, EventSetup const&) {
   std::cout << indention_ << indention_ << " finished begin lumi:" << std::endl;
}

void 
Tracer::prePathBeginLumi(std::string const& iName) {
  std::cout << indention_ << indention_ << indention_ << " processing path for begin lumi:" << iName << std::endl;
}
void 
Tracer::postPathBeginLumi(std::string const& iName, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " finished path for begin lumi:" << iName << std::endl;
}

void 
Tracer::preModuleBeginLumi(ModuleDescription const& iDescription) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for begin lumi:" << iDescription.moduleLabel() << std::endl;
}
void 
Tracer::postModuleBeginLumi(ModuleDescription const& iDescription) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   
   std::cout << " finished for begin lumi:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preEndLumi(LuminosityBlockID const& iID, Timestamp const& iTime) {
   depth_ = 0;
   std::cout << indention_ << indention_ << " processing end lumi:" << iID << " time:" << iTime.value() << std::endl;
}
void 
Tracer::postEndLumi(LuminosityBlock const&, EventSetup const&) {
   std::cout << indention_ << indention_ << " finished end lumi:" << std::endl;
}

void 
Tracer::prePathEndLumi(std::string const& iName) {
  std::cout << indention_ << indention_ << indention_ << " processing path for end lumi:" << iName << std::endl;
}

void 
Tracer::postPathEndLumi(std::string const& iName, HLTPathStatus const&) {
  std::cout << indention_ << indention_ << indention_ << " finished path for end lumi:" << iName << std::endl;
}

void 
Tracer::preModuleEndLumi(ModuleDescription const& iDescription) {
   ++depth_;
   std::cout << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   std::cout << " module for end lumi:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::postModuleEndLumi(ModuleDescription const& iDescription) {
   --depth_;
   std::cout << indention_ << indention_ << indention_ << indention_;
   for(unsigned int depth = 0; depth !=depth_; ++depth) {
      std::cout << indention_;
   }
   
   std::cout << " finished for end lumi:" << iDescription.moduleLabel() << std::endl;
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

void 
Tracer::preModuleConstruction(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " constructing module:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::postModuleConstruction(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " construction finished:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preModuleBeginJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " beginJob module:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::postModuleBeginJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " beginJob finished:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::preModuleEndJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " endJob module:" << iDescription.moduleLabel() << std::endl;
}

void 
Tracer::postModuleEndJob(ModuleDescription const& iDescription) {
  std::cout << indention_;
  std::cout << " endJob finished:" << iDescription.moduleLabel() << std::endl;
}

//
// const member functions
//

//
// static member functions
//

