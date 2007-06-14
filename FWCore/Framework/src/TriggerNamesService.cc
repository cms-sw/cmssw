// -*- C++ -*-
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id: TriggerNamesService.cc,v 1.7 2007/03/04 06:10:25 wmtan Exp $
//

#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Utilities/interface/Exception.h"


namespace edm {
  namespace service {

    typedef std::vector<std::string> vstring;

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

    TriggerNamesService::TriggerNamesService(const ParameterSet& pset) :
      process_name_(pset.getParameter<std::string>("@process_name")),
      trig_pset_(getTrigPSetFunc(pset)),
      trignames_(trig_pset_.getUntrackedParameter<vstring>("@trigger_paths")),
      trigpos_(),
      pathnames_(trig_pset_.getParameter<vstring>("@paths")),
      pathpos_(),
      end_names_(trig_pset_.getParameter<vstring>("@end_paths")),
      end_pos_(),
      modulenames_(),
      wantSummary_(),
      makeTriggerResults_()
    {
      ParameterSet defopts;
      ParameterSet opts = 
	pset.getUntrackedParameter<ParameterSet>("options", defopts);
      wantSummary_ =
	opts.getUntrackedParameter("wantSummary",false);
      makeTriggerResults_ = 
	opts.getUntrackedParameter("makeTriggerResults",false);

      loadPosMap(trigpos_,trignames_);
      loadPosMap(pathpos_,pathnames_);
      loadPosMap(end_pos_,end_names_);

      const unsigned int n(trignames_.size());
      for(unsigned int i=0;i!=n;++i) {
        modulenames_.push_back(pset.getParameter<vstring>(trignames_[i]));
      }
    }

    TriggerNamesService::~TriggerNamesService()
    {
    }

    TriggerNamesService::Bools TriggerNamesService::getBitMask(const Strings& names) const
    {
      Bools bv(names.size());
      Strings::const_iterator i(names.begin()),e(names.end());
      for(;i!=e;++i)
	{
	  PosMap::const_iterator ipos(trigpos_.find(*i));
	  if(ipos!=trigpos_.end())
	    bv[ipos->second] = true;
	  else
	    throw cms::Exception("NotFound")
	      << "Path name " << *i << " not found in trigger paths\n";
	}
      return bv;
    }
  }
}
