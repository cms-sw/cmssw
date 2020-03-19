#include <iostream>
#include <typeinfo>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"

// make sure PDFSET is pulled in when linking against the
// archive lhapdf library.
extern "C" void pdfset_(void);
__attribute__((visibility("hidden"))) void dummy() { pdfset_(); }
// implementation for the Fortran callbacks from Pythia6/Herwig6

void gen::upinit_() { FortranInstance::getInstance<FortranInstance>()->upInit(); }

void gen::upevnt_() { FortranInstance::getInstance<FortranInstance>()->upEvnt(); }

void gen::upveto_(int *veto) { *veto = FortranInstance::getInstance<FortranInstance>()->upVeto(); }

// static FortranInstance members;

gen::FortranInstance *gen::FortranInstance::currentInstance = nullptr;

const std::string gen::FortranInstance::kFortranInstance = "FortranInstance";

// FortranInstance methods

gen::FortranInstance::~FortranInstance() noexcept(false) {
  if (currentInstance == this) {
    edm::LogWarning("ReentrancyProblem") << edm::friendlyname::friendlyName(typeid(*this).name())
                                         << " destroyed while it was the "
                                            "current active instance."
                                         << std::endl;
    currentInstance = nullptr;
  }
}

// FortranInstance current instance tracking

void gen::FortranInstance::enter() {
  // we should add a boost::mutex here if we care about being
  // multithread-safe
  if (currentInstance && currentInstance != this)
    throw edm::Exception(edm::errors::LogicError) << edm::friendlyname::friendlyName(typeid(*this).name())
                                                  << "::enter() called from a different "
                                                     "instance while an instance was already active."
                                                  << std::endl;

  if (!currentInstance && instanceNesting != 0)
    throw edm::Exception(edm::errors::LogicError) << edm::friendlyname::friendlyName(typeid(*this).name())
                                                  << "::enter() called on an empty "
                                                     "instance, but instance counter is nonzero."
                                                  << std::endl;

  currentInstance = this;
  instanceNesting++;
}

void gen::FortranInstance::leave() {
  if (!currentInstance)
    throw edm::Exception(edm::errors::LogicError) << edm::friendlyname::friendlyName(typeid(*this).name())
                                                  << "::leave() called without an "
                                                     "active instance."
                                                  << std::endl;
  else if (currentInstance != this)
    throw edm::Exception(edm::errors::LogicError) << edm::friendlyname::friendlyName(typeid(*this).name())
                                                  << "::leave() called from a "
                                                     "different instance."
                                                  << std::endl;
  else if (instanceNesting <= 0)
    throw edm::Exception(edm::errors::LogicError) << edm::friendlyname::friendlyName(typeid(*this).name())
                                                  << "::leave() called with a "
                                                     "nesting level of zero."
                                                  << std::endl;

  if (--instanceNesting == 0)
    currentInstance = nullptr;
}

void gen::FortranInstance::throwMissingInstance() {
  throw edm::Exception(edm::errors::LogicError) << "FortranInstance::getInstance() called from "
                                                   "a Fortran context, but no current instance "
                                                   "has been registered."
                                                << std::endl;
}

// Herwig callback stubs

void gen::FortranInstance::upInit() {
  throw cms::Exception("UnimplementedCallback") << edm::friendlyname::friendlyName(typeid(*this).name())
                                                << "::upInit() stub called. "
                                                   "If user process needs to be generated, please derive "
                                                   "and implement the upInit() method."
                                                << std::endl;
}

void gen::FortranInstance::upEvnt() {
  throw cms::Exception("UnimplementedCallback") << edm::friendlyname::friendlyName(typeid(*this).name())
                                                << "::upEvnt() stub called. "
                                                   "If user process needs to be generated, please derive "
                                                   "and implement the upEvnt() method."
                                                << std::endl;
}

bool gen::FortranInstance::upVeto() { return false; }
