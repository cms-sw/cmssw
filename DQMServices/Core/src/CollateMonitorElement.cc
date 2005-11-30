#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;

CollateMonitorElement::CollateMonitorElement(const string & name, 
					     const string title, 
					     const string pathname)
{
  searchStrings.clear(); contents_.clear(); canUse_ = false;
  numUsed_ = 0; cname_ = name; ctitle_ = title; cpathname_ = pathname;
}


CollateMonitorElement::~CollateMonitorElement() 
{
}

// add <search_string> to summary ME; 
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void CollateMonitorElement::add2search(const string & search_string)
{
  searchStrings.insert(search_string);
}

// look for all ME matching <search_string> in <look_here>; 
// if found, add to contents_
void CollateMonitorElement::scanContents(const string & search_string, const 
					 global_map & look_here)
{
  for(cglob_it path = look_here.begin(); 
      path != look_here.end(); ++path)
    { // loop over pathnames of global_map
      
      string pathname = path->first;
      for(cME_it file = path->second->begin(); 
	  file != path->second->end(); ++file)
	{ // loop over files of <pathname>
	  
	  string fullname = getUnixName(pathname, file->first);
	  
	  if(matchit(fullname, search_string))
	    { // this is a match!
	      MonitorElement * me = (MonitorElement *)file->second;
	      
	      if(addIt(me, pathname, file->first))
		{
		  // check if we need to define histogram/profile
		  if(!canUse_)
		    createCollateBase(me);
		}
	      
	    } // this is a match!
	  
	} // loop over files of <pathname>
      
    } // loop over pathnames of global_map
  
}

// add <search_string> to cme's contents; look for match in global_map
void CollateMonitorElement::add(const string & search_string, const global_map & 
				look_here)
{
  add2search(search_string);
  scanContents(search_string, look_here);
}

