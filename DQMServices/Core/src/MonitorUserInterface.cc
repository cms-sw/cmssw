#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;
using namespace dqm::qtests;

/* Use the default constructor for running in standalone mode (ie. without
   sources or collectors); if flag=true, client will accept downstream connections
*/
MonitorUserInterface::MonitorUserInterface()
{
  collate_mes.clear(); numUpdates_ = 0; bei = 0;
}

MonitorUserInterface::~MonitorUserInterface(void)
{
  for(scmeIt it = collate_mes.begin(); it != collate_mes.end(); ++it)
    delete (*it);

  collate_mes.clear();
}

// to be used by methods subscribe (add=true) and unsubscribe (add=false)
// <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::subscribe_base(const string & subsc_request, bool add)
{
  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests; requests.clear();

  cglob_it start, end, parent_dir;
  getSubRange<global_map>(subsc_request, bei->own.global_, 
			  start, end, parent_dir);
  
  for(cglob_it path = start; path != end; ++path)
    // loop over pathnames of global_map
    subscribe_base(subsc_request, add, requests, path);

  if(parent_dir != bei->own.global_.end())
    subscribe_base(subsc_request, add, requests, parent_dir);

  // (un)subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, add);
  
}

// like subscribe_base above, for one path only
void MonitorUserInterface::subscribe_base(const string & subsc_request, bool add,
					  vector<string> & requests, 
					  cglob_it path)
{
  string new_request;
  for(cME_it file = path->second->begin(); file != path->second->end(); 
      ++file)
    { // loop over files in directory <path->first>
	  
      // "subscribe" should loop over "monitorable";
      // "unsubscribe" should loop over "contents" (only non-null MEs)
      
      // if unsubscribing, skip null monitoring elements
      if(!add && !file->second)continue;
      
      string fullname = getUnixName(path->first, file->first);
	  
      if(matchit(fullname, subsc_request))
	{ // this is a match!
	  
	  if(!new_request.empty())new_request += ",";
	  new_request += file->first;
	  
	} // this is a match!
	  
    }  // loop over files in directory <path->first>
      
  if(!new_request.empty())
    requests.push_back(path->first + ":" + new_request);
  
}

// subscription request; format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::subscribe(const string & subsc_request)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;
  subscribe_base(subsc_request, true);
}

// unsubscription request; format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::unsubscribe(const string & subsc_request)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;
  subscribe_base(subsc_request, false);
}

// similar to method subscribe; use only additions to monitorable in last cycle
void MonitorUserInterface::subscribeNew(const string & subsc_request)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  vector<string> put_here; put_here.clear();
  bei->getAddedMonitorable(put_here);
  
  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests; requests.clear();

  DirFormat dir;
  for(cvIt mon = put_here.begin(); mon != put_here.end(); ++mon)
    { // loop over added monitorable's pathnames
      if(!unpackDirFormat(*mon, dir))
	continue;

      string new_request;
      for(cvIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
	{ // loop over files in directory

	  string fullname = getUnixName(dir.dir_path, *it);
	  if(matchit(fullname, subsc_request))
	    { // this is a match!

	      if(!new_request.empty())new_request += ",";
	      new_request += *it;
	      
	    } // this is a match!

	}  // loop over files in directory

      if(!new_request.empty())
	requests.push_back(dir.dir_path + ":" + new_request);

    } // loop over added monitorable's pathnames

  // subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, true);


}

// set(unset) subscription if add=true(false)
// string format: <dir pathname>:<h1>,<h2>,...
void MonitorUserInterface::finishSubscription(const vector<string> & monit, 
					      bool add)
{
  LockMutex a(bei->requests.mutex);

  for(cvIt it = monit.begin(); it != monit.end(); ++it){

    // (un)subscribe to monitorable
    if(add)
      bei->requests.toAdd.push_back(*it);
    else
      bei->requests.toRemove.push_back(*it);
  }
}

// save dir_fullpath with monitoring objects into root file <filename>;
// include quality test results with status >= minimum_status 
// (defined in Core/interface/QTestStatus.h);
// if dir_fullpath="", save full monitoring structure
void MonitorUserInterface::save(string filename, string dir_fullpath,
				int minimum_status) const
{
  bei->save(filename, dir_fullpath, minimum_status);
}

// draw all monitoring objects
void MonitorUserInterface::drawAll(void) const
{
  bei->drawAll();
}

// reset ME contents (does not permanently reset contents)
void MonitorUserInterface::softReset(MonitorElement * me)
{
  bei->softReset(me);
}

// reverts action of softReset
void MonitorUserInterface::disableSoftReset(MonitorElement * me)
{
  bei->disableSoftReset(me);
}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
void MonitorUserInterface::setAccumulate(MonitorElement * me, bool flag)
{
  bei->setAccumulate(me, flag);
}

// add <search_string> to summary ME; 
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::add(CollateMonitorElement * cme, 
			       const string & search_string) const
{
  if(search_string.empty())
    return;

  if(!cme)
    {
      cerr << " *** Cannot use " << search_string 
	   << " with null CollateMonitorElement! " << endl;
      return;
    }

  cme->add(search_string, bei->own.global_);
}

// new MEs have been added; check if need to update collate-MEs
void MonitorUserInterface::checkAddedContents(void)
{
  for(scmeIt cme = collate_mes.begin(); cme != collate_mes.end(); ++cme){
    // loop over collate-MEs
    for(csIt search_string = (*cme)->searchStrings.begin(); 
	search_string != (*cme)->searchStrings.end(); ++search_string){
      // loop over search-strings for CME
      
      cmonit_it start, end, parent_dir;
      getSubRange<monit_map>(*search_string, bei->addedContents, 
			     start, end, parent_dir);
      
      for(cmonit_it path = start; path != end; ++path)
	// loop over all pathnames of added contents
	checkAddedContents(*search_string, cme, path);
      
      if(parent_dir != bei->addedContents.end())
	checkAddedContents(*search_string, cme, parent_dir);
      
    } // loop over search-strings for CME
	  
  } // loop over collate-MEs

}

// save as above for given search_string and path
void MonitorUserInterface::checkAddedContents(const string & search_string, 
					      scmeIt & cme,
					      cmonit_it & path)
{
  for(csIt it = path->second.begin(); it!= path->second.end(); ++it){
    // loop over all added MEs
    
    // get unix-like filename
    string fullname = getUnixName(path->first, *it);

    if(matchit(fullname, search_string)){
      // this is a match!
      MonitorElement* me = bei->findObject(*it, path->first);
      bool didIt = (*cme)->addIt(me, path->first, *it);
      
      if(didIt && !(*cme)->canUse_)
	(*cme)->createCollateBase(me);
      
    } // this is a match!
		      
  }  // loop over all added MEs

}

// do calculations for all collate MEs; come here at end of monitoring cycle)
void MonitorUserInterface::doSummary(void)
{
  if(!bei->addedContents.empty())
    checkAddedContents();
  
  for(scmeIt it = collate_mes.begin(); it != collate_mes.end(); ++it)
    {
      (*it)->summary();
    }
}

// this is the "main" loop where we receive monitoring/send subscription requests;
// if client acts as server, method runQTests is also sending monitoring & 
// test results to clients downstream;
// returns success flag
bool MonitorUserInterface::update(void)
{

  // retrieval of monitoring, sending of subscription requests/cancellations,
  // calculation of "collate"-type Monitoring Elements;
  bool ret = doMonitoring();

  // Run quality tests (and determine updated contents);
  // Method is overloaded if client acts as server to other clients downstream
  runQTests();

  return ret;
}

// remove CollateMonitorElement
void MonitorUserInterface::removeCollate(CollateMonitorElement * cme)
{
  if(!cme)
    {
      cerr << " *** Attempt to remove null CollateMonitorElement ! " << endl;
      return;
    }

  collate_mes.erase(cme);
  delete cme;
}
