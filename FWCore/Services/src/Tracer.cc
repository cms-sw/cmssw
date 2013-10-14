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

#include "FWCore/Services/src/Tracer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

#include <iostream>

using namespace edm::service;

Tracer::Tracer(ParameterSet const& iPS, ActivityRegistry&iRegistry) :
  indention_(iPS.getUntrackedParameter<std::string>("indention")),
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

void
Tracer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("indention", "++")->setComment("Prefix characters for output. The characters are repeated to form the indentation.");
  desc.addUntracked<std::string>("dumpContextForLabel", "")->setComment("Prints context information to cout for the module transitions associated with this module label");
  desc.addUntracked<bool>("dumpNonModuleContext", false)->setComment("Prints context information to cout for the transitions not associated with any module label");
  descriptions.add("Tracer", desc);
  descriptions.setComment("This service prints each phase the framework is processing, e.g. constructing a module,running a module, etc.");
}

void 
Tracer::postBeginJob() {
  LogAbsolute("Tracer") << indention_ << " finished: begin job";
}

void 
Tracer::postEndJob() {
  LogAbsolute("Tracer") << indention_ << " finished: end job";
}

void
Tracer::preSource() {
  LogAbsolute("Tracer") << indention_ << indention_ << " starting: source event";
}

void
Tracer::postSource() {
  LogAbsolute("Tracer") << indention_ << indention_ << " finished: source event";
}

void
Tracer::preSourceLumi() {
  LogAbsolute("Tracer") << indention_ << indention_ << " starting: source lumi";
}

void
Tracer::postSourceLumi () {
  LogAbsolute("Tracer") << indention_ << indention_ << " finished: source lumi";
}

void
Tracer::preSourceRun() {
  LogAbsolute("Tracer") << indention_ << indention_ << " starting: source run";
}

void
Tracer::postSourceRun () {
  LogAbsolute("Tracer") << indention_ << indention_ << " finished: source run";
}

void
Tracer::preOpenFile(std::string const& lfn, bool b) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: open input file: lfn = " << lfn;
  if(dumpNonModuleContext_) out << " usedFallBack = " << b;
}

void
Tracer::postOpenFile (std::string const& lfn, bool b) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: open input file: lfn = " << lfn;
  if(dumpNonModuleContext_) out << " usedFallBack = " << b;
}

void
Tracer::preCloseFile(std::string const & lfn, bool b) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: close input file: lfn = " << lfn;
  if(dumpNonModuleContext_) out << " usedFallBack = " << b;
}
void
Tracer::postCloseFile (std::string const& lfn, bool b) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: close input file: lfn = " << lfn;
  if(dumpNonModuleContext_) out << " usedFallBack = " << b;
}

void
Tracer::preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: begin stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: begin stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: end stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: end stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preGlobalBeginRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: global begin run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::postGlobalBeginRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: global begin run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::preGlobalEndRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: global end run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::postGlobalEndRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: global end run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::preStreamBeginRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: begin run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::postStreamBeginRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: begin run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::preStreamEndRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: end run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::postStreamEndRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: end run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::preGlobalBeginLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: global begin lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::postGlobalBeginLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: global begin lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::preGlobalEndLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: global end lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::postGlobalEndLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: global end lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void
Tracer::preStreamBeginLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: begin lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::postStreamBeginLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: begin lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::preStreamEndLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: end lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::postStreamEndLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: end lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::preEvent(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: processing event : stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " event = " << sc.eventID().event() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::postEvent(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: processing event : stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " lumi = " << sc.eventID().luminosityBlock() << " event = " << sc.eventID().event() << " time = " << sc.timestamp().value();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void
Tracer::prePathEvent(StreamContext const& sc, PathContext const& pc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << indention_ << " starting: processing path '" << pc.pathName() << "' : stream = " << sc.streamID();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
    out << pc;
  }
}

void
Tracer::postPathEvent(StreamContext const& sc, PathContext const& pc, HLTPathStatus const& hlts) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << indention_ << " finished: processing path '" << pc.pathName() << "' : stream = " << sc.streamID();
  if(dumpNonModuleContext_) {
    out << "\n" << sc;
    out << pc;
  }
}

void 
Tracer::preModuleConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: constructing module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::postModuleConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: constructing module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::preModuleBeginJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_;
  out << " starting: begin job for module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::postModuleBeginJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_;
  out << " finished: begin job for module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::preModuleEndJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_;
  out << " starting: end job for module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::postModuleEndJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_;
  out << " finished: end job for module with label '" << desc.moduleLabel() << "'";
  if(dumpContextForLabel_ == desc.moduleLabel()) {
    out << "\n" << desc;
  }
}

void 
Tracer::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 4;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void 
Tracer::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 4;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}


void
Tracer::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: begin run for module: stream = " << sc.streamID() <<  " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: begin run for module: stream = " << sc.streamID() <<  " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: end run for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: end run for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: begin lumi for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: begin lumi for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: end lumi for module: stream = " << sc.streamID() << " label = '"<< mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: end lumi for module: stream = " << sc.streamID() << " label = '"<< mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void
Tracer::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for(unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "'";
  if(dumpContextForLabel_ == mcc.moduleDescription()->moduleLabel()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void
Tracer::preSourceConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_;
  out << " starting: constructing source: " << desc.moduleName();
  if(dumpNonModuleContext_) {
    out << "\n" << desc;
  }
}

void
Tracer::postSourceConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << indention_;
  out << " finished: constructing source: " << desc.moduleName();
  if(dumpNonModuleContext_) {
    out << "\n" << desc;
  }
}
