// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//

#include "FWCore/Services/src/Profiling.h"
#include "FWCore/Services/src/SimpleProfiler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  namespace service {

    SimpleProfiling::SimpleProfiling(ParameterSet const&,
                                     ActivityRegistry&iRegistry) {
      iRegistry.watchPostBeginJob(this, &SimpleProfiling::postBeginJob);
      iRegistry.watchPostEndJob(this, &SimpleProfiling::postEndJob);
    }


    SimpleProfiling::~SimpleProfiling() {
    }

    void SimpleProfiling::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      descriptions.add("SimpleProfiling", desc);
    }

    void SimpleProfiling::postBeginJob() {
      LogInfo("SimpleProfiling")
        << "Simple profiling activated.\n";

      SimpleProfiler::instance()->start();
    }

    void SimpleProfiling::postEndJob() {
      pid_t pid = getpid();

      LogInfo("SimpleProfiling")
        << "Simple profiling stopping.\n"
        << "You should find three files containing profiling\n"
        << "information gathered from program counter\n"
        << "samples collected while your job was running.\n"
        << "\tA) profdata_" << pid << "_names\n"
        << "\tB) profdata_" << pid << "_paths\n"
        << "\tC) profdata_" << pid << "_totals\n"
        << "\n"
        << "A) contains names of function hit count.  You may want\n"
        << "   to pass this through c++filt since the name is still mangled\n"
        << " columns:\n"
        << "\t1) function ID\n"
        << "\t2) function address\n"
        << "\t3) samples within this function only (leaf count)\n"
        << "\t4) samples within this function and children (with recusion)\n"
        << "\t5) samples within this function and children\n"
        << "\t6) fraction of samples within this function only\n"
        << "\t7) fraction of samples within this path (with recusion)\n"
        << "\t8) function name (mangled)\n"
        << "The file is sorted by column (3) so most-hit functions\n"
        << "are at the top\n"
        << "\n"
        << "B) contains all the unique call paths traversed in this job\n"
        << "  columns:\n"
        << "\t1) path ID\n"
        << "\t2) total times this unique path was observed (samples in leaf)\n"
        << "\t3) the path using function IDs (main towards the left)\n"
        << "\n"
        << "C) contains summary of the total number of samples collected\n"
        << "\n";

      SimpleProfiler::instance()->stop();
    }
  }
}
