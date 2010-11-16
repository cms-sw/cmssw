#include "DQMServices/Core/interface/Standalone.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "classlib/utils/DebugAids.h"
#include "classlib/utils/Signal.h"
#include "TROOT.h"
#include <iostream>
#include <errno.h>

#ifndef _NSIG
#define _NSIG NSIG
#endif

struct MEInfo
{
  int		runnr;
  std::string	system;
  std::string	category;
  std::string	name;
  std::string	data;
  char		style;
};

static const int FATAL_OPTS = (lat::Signal::FATAL_DEFAULT
			       & ~(lat::Signal::FATAL_ON_INT
				   | lat::Signal::FATAL_ON_QUIT
				   | lat::Signal::FATAL_DUMP_CORE));

// -------------------------------------------------------------------
// Always abort on assertion failure.
static char
onAssertFail(const char *message)
{
  std::cout.flush(); fflush(stdout);
  std::cerr.flush(); fflush(stderr);
  std::cerr << message << "ABORTING\n";
  return 'a';
}

static const char *
kindToName(MonitorElement::Kind kind)
{
  switch (kind)
  {
  case MonitorElement::DQM_KIND_INVALID:    return "INVALID";
  case MonitorElement::DQM_KIND_INT:        return "INT";
  case MonitorElement::DQM_KIND_REAL:       return "REAL";
  case MonitorElement::DQM_KIND_STRING:     return "STRING";
  case MonitorElement::DQM_KIND_TH1F:       return "TH1F";
  case MonitorElement::DQM_KIND_TH1S:       return "TH1S";
  case MonitorElement::DQM_KIND_TH1D:       return "TH1D";
  case MonitorElement::DQM_KIND_TH2F:       return "TH2F";
  case MonitorElement::DQM_KIND_TH2S:       return "TH2S";
  case MonitorElement::DQM_KIND_TH2D:       return "TH2D";
  case MonitorElement::DQM_KIND_TH3F:       return "TH3F";
  case MonitorElement::DQM_KIND_TPROFILE:   return "TPROFILE";
  case MonitorElement::DQM_KIND_TPROFILE2D: return "TPROFILE2D";
  default:				    assert(false); return "INTERNAL_ERROR";
  }
}

// -------------------------------------------------------------------
// Extract standard parameters from the DQM data.
static void
getMEInfo(DQMStore &store, MonitorElement &me, MEInfo &info)
{
  info.style = 'U';
  info.runnr = -1;
  info.system.clear();
  info.category.clear();
  info.name.clear();
  info.data.clear();

  switch (me.kind())
  {
  case MonitorElement::DQM_KIND_INT:
  case MonitorElement::DQM_KIND_REAL:
  case MonitorElement::DQM_KIND_STRING:
    info.data = me.tagString();
    break;

  default:
    break;
  }

  // If it's a reference, strip reference part out.
  bool isref = false;
  std::string name = me.getFullname();
  if (name.size() > 10 && name.compare(0, 10, "Reference/") == 0)
  {
    name.erase(0, 10);
    isref = true;
  }

  // First check for leading "Run XYZ/System/Category/Name".
  size_t slash, sys, cat;
  if (name.compare(0, 4, "Run ") == 0
      && (slash = name.find('/')) < name.size()-1
      && (sys = name.find('/', slash+1)) < name.size()-1
      && (cat = name.find('/', sys+1)) < name.size()-1)
  {
    info.system.append(name, slash+1, sys-slash-1);
    info.category.append(name, sys+1, cat-sys-1);
    info.name.append(name, cat+1, std::string::npos);

    errno = 0;
    char *end = 0;
    info.runnr = strtol(name.c_str()+4, &end, 10);
    if (errno != 0 || !end || *end != '/')
      info.runnr = -1;

    info.style = isref ? 'c' : 'C';
    return;
  }

  // Try "System/EventInfo/iRun" or just "System/Name"
  if ((slash = name.find('/')) != std::string::npos && slash > 0)
  {
    std::string system(name, 0, slash);
    info.system = system;
    info.category.clear();
    info.name.append(name, slash+1, std::string::npos);
    if (MonitorElement *runnr = store.get(system + "/EventInfo/iRun"))
      info.runnr = runnr->getIntValue();
    else
      info.runnr = -1;

    info.style = isref ? 's' : 'S';
    return;
  }

  // Otherwise use the defaults but fill in the name.
  info.name = name;
}

std::string
hexlify(const std::string &x)
{
  std::string result;
  result.reserve(2*x.size() + 1);
  for (size_t i = 0, e = x.size(); i != e; ++i)
  {
    char buf[3];
    sprintf(buf, "%02x", (unsigned) (unsigned char) x[i]);
    result += buf[0];
    result += buf[1];
  }
  return result;
}

// -------------------------------------------------------------------
// Main program.
int main(int argc, char **argv)
{
  // Install base debugging support.
  lat::DebugAids::failHook(&onAssertFail);
  lat::Signal::handleFatal(argv[0], IOFD_INVALID, 0, 0, FATAL_OPTS);

  // Re-capture signals from ROOT after ROOT has initialised.
  ROOT::GetROOT();
  for (int sig = 1; sig < _NSIG; ++sig) lat::Signal::revert(sig);
  lat::Signal::handleFatal(argv[0], IOFD_INVALID, 0, 0, FATAL_OPTS);

  // Check command line arguments.
  int arg = 1;
  int bad = 0;
  std::string dataset;
  std::string step;
  for ( ; arg < argc; ++arg)
    if (arg < argc-1 && !strcmp(argv[arg], "--dataset"))
      dataset = argv[++arg];
    else if (arg < argc-1 && !strcmp(argv[arg], "--step"))
      step = argv[++arg];
    else if (argv[arg][0] == '-')
      ++bad;
    else
      break;

  if (bad)
  {
    std::cerr << "Usage: " << argv[0]
	      << " [--dataset NAME]"
	      << " [--step NAME]"
	      << " FILE...\n";
    return 1;
  }

  // Process each file given as argument.
  edm::ParameterSet emptyps;
  std::vector<edm::ParameterSet> emptyset;
  edm::ServiceToken services(edm::ServiceRegistry::createSet(emptyset));	 
  edm::ServiceRegistry::Operate operate(services);	 
  DQMStore store(emptyps);	
  MEInfo info;
  for ( ; arg < argc; ++arg)
  {
    try
    {
      // Read in the file.
      store.open(argv[arg]);
      std::cout << "FILE NAME='" << argv[arg] << "'\n";

      // Dump info about the monitor elements.
      std::vector<MonitorElement *> mes = store.getAllContents("");
      for (size_t m = 0, e = mes.size(); m < e; ++m)
      {
        MonitorElement &me = *mes[m];
        getMEInfo(store, me, info);
        std::cout << "ME STYLE=" << info.style
		  << " RUN=" << info.runnr
		  << " DATASET='" << dataset
		  << "' STEP='" << step
		  << "' SYSTEM='" << info.system
		  << "' CATEGORY='" << info.category
		  << "' KIND='" << kindToName(me.kind())
		  << "' TAG=" << me.getTag()
		  << " FLAGS=0x" << std::hex << me.flags() << std::dec
		  << " NAME='" << info.name
		  << "' DATA='" << hexlify(info.data)
		  << "'\n";
      }
    }
    catch (std::exception &e)
    {
      std::cerr << "*** FAILED TO READ FILE " << argv[arg] << ":\n"
		<< e.what() << std::endl;
    }

    // Now clear the DQM store for the next file.
    store.rmdir("");
  }

  return 0;
}
