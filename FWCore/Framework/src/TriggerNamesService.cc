// -*- C++ -*-
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id: TriggerNamesService.cc,v 1.3 2006/02/08 00:44:25 wmtan Exp $
//

#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm {
  namespace service {

    typedef vector<string> vstring;

    void checkIfSubset(const vstring& in_all, const vstring& in_sub)
    {
      vstring all(in_all), sub(in_sub), result;
      sort(all.begin(),all.end());
      sort(sub.begin(),sub.end());
      set_intersection(all.begin(),all.end(),
		       sub.begin(),sub.end(),
		       back_inserter(result));

      if(result.size() != sub.size())
	throw cms::Exception("TriggerPaths")
	  << "Specified listOfTriggers is not a subset of the available paths\n";
    }

    ParameterSet getTrigPSetFunc(ParameterSet const& proc_pset)
    {
      ParameterSet rc = 
	proc_pset.getUntrackedParameter<ParameterSet>("@trigger_paths");
      bool want_results = false;
      // default for trigger paths is all the paths
      vstring allpaths = rc.getParameter<vstring>("@paths");

      // the value depends on options and value of listOfTriggers
      try {
	ParameterSet defopts;
        ParameterSet opts = 
	  proc_pset.getUntrackedParameter<ParameterSet>("options", defopts);
	want_results =
	  opts.getUntrackedParameter<bool>("makeTriggerResults",false);

	// if makeTriggerResults is true, then listOfTriggers must be given

	if(want_results) {
	  vstring tmppaths = opts.getParameter<vstring>("listOfTriggers");

	  // verify that all the names in allpaths are a subset of
	  // the names currently in allpaths (all the names)

	  if(!tmppaths.empty() && tmppaths[0] == "*") {
	    // leave as full list
	  } else {
	    checkIfSubset(allpaths, tmppaths);
	    allpaths.swap(tmppaths);
	  }
	}
      }
      catch(edm::Exception& e) {
      }

      rc.addUntrackedParameter<vstring>("@trigger_paths",allpaths);
      return rc;
    }

    // note: names_ comes from "@trigger_paths", the parameter that has
    // the options applied to it.

    TriggerNamesService::TriggerNamesService(const ParameterSet& pset):
      process_name_(pset.getParameter<string>("@process_name")),
      trig_pset_(getTrigPSetFunc(pset)),
      names_(trig_pset_.getUntrackedParameter<vstring>("@trigger_paths")),
      paths_(trig_pset_.getParameter<vstring>("@paths")),
      end_paths_(trig_pset_.getParameter<vstring>("@end_paths"))
    {
      ParameterSet defopts;
      ParameterSet opts = 
	pset.getUntrackedParameter<ParameterSet>("options", defopts);
      wantSummary_ =
	opts.getUntrackedParameter("wantSummary",false);
      makeTriggerResults_ = 
	opts.getUntrackedParameter("makeTriggerResults",false);

      for(unsigned int i=0;i<names_.size();++i) pos_[names_[i]] = i;
    }


    TriggerNamesService::~TriggerNamesService()
    {
    }

    TriggerNamesService::BitVector TriggerNamesService::getBitMask(const Strings& names) const
    {
      BitVector bv(names_.size());
      Strings::const_iterator i(names.begin()),e(names.end());
      for(;i!=e;++i)
	{
	  PosMap::const_iterator ipos(pos_.find(*i));
	  if(ipos!=pos_.end())
	    bv[ipos->second] = true;
	  else
	    throw cms::Exception("NotFound")
	      << "Path name " << *i << " not found in trigger path\n";
	}

      return bv;
    }
  }
}
