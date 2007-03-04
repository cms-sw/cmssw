// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
// 
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id: Profiling.cc,v 1.1 2006/03/11 04:28:21 jbk Exp $
//

#include "FWCore/Services/src/Profiling.h"
#include "FWCore/Services/src/SimpleProfiler.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

namespace edm {
  namespace service {

    SimpleProfiling::SimpleProfiling(const ParameterSet& iPS,
				     ActivityRegistry&iRegistry)
    {
      iRegistry.watchPostBeginJob(this,&SimpleProfiling::postBeginJob);
      iRegistry.watchPostEndJob(this,&SimpleProfiling::postEndJob);
    }


    SimpleProfiling::~SimpleProfiling()
    {
    }

    void SimpleProfiling::postBeginJob()
    {
      LogInfo("SimpleProfiling")
	<< "Simple profiling activated.\n";

      SimpleProfiler::instance()->start();      
    }

    void SimpleProfiling::postEndJob()
    {
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
	<< "\t3) function name (mangled)\n"
	<< "\t4) samples within this function only (leaf count)\n"
	<< "\t5) samples within this function and children (with recusion)\n"
	<< "\t6) samples within this function and children\n"
	<< "\t7) fraction of samples within this function only\n"
	<< "\t8) fraction of samples within this path (with recusion)\n"
	<< "The file is sorted by column (4) so most-hit functions\n"
	<< "are at the top\n"
	<< "\n"
	<< "B) contains all the unique call paths traversed in this job\n"
	<< "  columns:\n"
	<< "\t1) path ID\n"
	<< "\t2) total times this unique path was used (samples in leaf)\n"
	<< "\t3) the path using function IDs (main towards the left)\n"
	<< "\n"
	<< "C) contains summary of the total number of samples collected\n"
	<< "\n";

      SimpleProfiler::instance()->stop();
    }
  }
}
