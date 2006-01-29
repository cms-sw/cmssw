// -*- C++ -*-
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id$
//

#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm {
  namespace service {

    TriggerNamesService::TriggerNamesService(const ParameterSet& pset)
    {
      names_ = pset.getParameter<Strings>("@paths");
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
