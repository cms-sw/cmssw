//
//  ThreadsInfo.cc
//  CMSSW
//
//  Created by Chris Jones on 7/24/20.
//
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  ThreadsInfo threadOptions(edm::ParameterSet const& pset) {
    // default values
    ThreadsInfo threadsInfo;

    // Note: it is important to not check the type or trackedness in
    // exists() call to ensure that the getUntrackedParameter() calls
    // will fail if the parameters have an incorrect type
    if (pset.exists("options")) {
      auto const& ops = pset.getUntrackedParameterSet("options");
      if (ops.exists("numberOfThreads")) {
        threadsInfo.nThreads_ = ops.getUntrackedParameter<unsigned int>("numberOfThreads");
      }
      if (ops.exists("sizeOfStackForThreadsInKB")) {
        threadsInfo.stackSize_ = ops.getUntrackedParameter<unsigned int>("sizeOfStackForThreadsInKB");
      }
    }
    return threadsInfo;
  }

  void setThreadOptions(ThreadsInfo const& threadsInfo, edm::ParameterSet& pset) {
    edm::ParameterSet newOp;
    if (pset.exists("options")) {
      newOp = pset.getUntrackedParameterSet("options");
    }
    newOp.addUntrackedParameter<unsigned int>("numberOfThreads", threadsInfo.nThreads_);
    newOp.addUntrackedParameter<unsigned int>("sizeOfStackForThreadsInKB", threadsInfo.stackSize_);
    pset.insertParameterSet(true, "options", edm::ParameterSetEntry(newOp, false));
  }
}  // namespace edm
