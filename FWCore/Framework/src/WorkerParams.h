#ifndef Framework_WorkerParams_h
#define Framework_WorkerParams_h

/** ----------------------

This struct is used to communication parameters into the worker factory.

---------------------- **/

#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edm
{
  class ProductRegistry;
  class ActionTable;

  struct WorkerParams
  {
    WorkerParams(): 
      procPset_(0), pset_(0),reg_(0),actions_(0),
      processName_(),releaseVersion_(),passID_() { }

    WorkerParams(ParameterSet const& procPset,
		 ParameterSet const& pset,
		 ProductRegistry& reg,
		 ActionTable& actions,
		 std::string const& processName,
		 std::string releaseVersion=getReleaseVersion(),
		 std::string passID=getPassID()):
      procPset_(&procPset),pset_(&pset),reg_(&reg),actions_(&actions),
      processName_(processName),releaseVersion_(releaseVersion),passID_(passID) { }

    ParameterSet const* procPset_;
    ParameterSet const* pset_;
    ProductRegistry* reg_;
    ActionTable* actions_;
    std::string processName_;
    ReleaseVersion releaseVersion_;
    PassID passID_;
  };
}

#endif
