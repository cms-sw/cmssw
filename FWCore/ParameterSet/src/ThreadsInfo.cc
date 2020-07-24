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

    if (pset.existsAs<edm::ParameterSet>("options", false)) {
      auto const& ops = pset.getUntrackedParameterSet("options");
      if (ops.existsAs<unsigned int>("numberOfThreads", false)) {
        threadsInfo.nThreads_ = ops.getUntrackedParameter<unsigned int>("numberOfThreads");
      }
      if (ops.existsAs<unsigned int>("sizeOfStackForThreadsInKB", false)) {
        threadsInfo.stackSize_ = ops.getUntrackedParameter<unsigned int>("sizeOfStackForThreadsInKB");
      }
    }
    return threadsInfo;
  }

  void setThreadOptions(ThreadsInfo const& threadsInfo, edm::ParameterSet& pset) {
    edm::ParameterSet newOp;
    if (pset.existsAs<edm::ParameterSet>("options", false)) {
      newOp = pset.getUntrackedParameterSet("options");
    }
    newOp.addUntrackedParameter<unsigned int>("numberOfThreads", threadsInfo.nThreads_);
    newOp.addUntrackedParameter<unsigned int>("sizeOfStackForThreadsInKB", threadsInfo.stackSize_);
    pset.insertParameterSet(true, "options", edm::ParameterSetEntry(newOp, false));
  }
}  // namespace edm
