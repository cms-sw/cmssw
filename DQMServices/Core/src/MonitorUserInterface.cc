#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;
using namespace dqm::qtests;

MonitorUserInterface::MonitorUserInterface(const string & hostname, int port_no, 
					   const string & client_name)
{
  collate_mes.clear(); numUpdates_ = 0;
}

MonitorUserInterface::~MonitorUserInterface(void)
{
  for(vector<CollateMonitorElement*>::iterator it = collate_mes.begin();
      it != collate_mes.end(); ++it)
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

  for(cglob_it path = bei->own.global_.begin(); 
      path != bei->own.global_.end(); ++path)
    { // loop over pathnames of global_map
      
      string new_request;

      for(cME_it file = path->second->begin(); 
	  file != path->second->end(); ++file)
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

    } // loop over pathnames of global_map

  // (un)subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, add);
  
}

// subscription request; format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::subscribe(const string & subsc_request)
{
  subscribe_base(subsc_request, true);
}

// unsubscription request; format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::unsubscribe(const string & subsc_request)
{
  subscribe_base(subsc_request, false);
}

// similar to method subscribe; use only additions to monitorable in last cycle
void MonitorUserInterface::subscribeNew(const string & subsc_request)
{
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
  for(cvIt it = monit.begin(); it != monit.end(); ++it){

    // (un)subscribe to monitorable
    if(add)
      bei->request2add.push_back(*it);
    else
      bei->request2remove.push_back(*it);
  }
}

// save structure with monitoring objects into root file
void MonitorUserInterface::save(const string & filename)
{
  bei->save(filename);
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

// if true: will subtract contents copied at "soft-reset" from now on
// if false: will NO longer subtract contents (default)
void MonitorUserInterface::enableSoftReset(MonitorElement * me, bool flag)
{
  bei->enableSoftReset(me, flag);
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
  if(!cme)
    {
      cerr << " *** Cannot use " << search_string 
	   << " with null CollateMonitorElement! " << endl;
    }
  else
    cme->add(search_string, bei->own.global_);
}

// new MEs have been added; check if need to update collate-MEs
void MonitorUserInterface::checkAddedContents(void)
{
  typedef vector<CollateMonitorElement *>::const_iterator It;
  for(It cme = collate_mes.begin(); cme != collate_mes.end(); ++cme){
    // loop over collate-MEs
    for(csIt search_string = (*cme)->searchStrings.begin(); 
	search_string != (*cme)->searchStrings.end(); ++search_string){
      // loop over search-strings for CME
      
      for(cmonit_it path = bei->addedContents.begin(); 
	  path != bei->addedContents.end(); ++path){
	// loop over all pathnames of added contents
	  
	for(csIt it = path->second.begin(); it!= path->second.end(); ++it){
	  // loop over all added MEs
	      
	  // get unix-like filename
	  string fullname = getUnixName(path->first, *it);
	  if(matchit(fullname, *search_string)){
	    // this is a match!
	    MonitorElement* me = bei->findObject(*it, path->first);
	    bool didIt = (*cme)->addIt(me, path->first, *it);
	    
	    if(didIt && !(*cme)->canUse_)
	      (*cme)->createCollateBase(me);
	    
	  } // this is a match!
		      
	}  // loop over all added MEs
	      
      } // loop over all pathnames of added contents
      

    } // loop over search-strings for CME
	  
  } // loop over collate-MEs

}

// do calculations for all collate MEs; come here at end of monitoring cycle)
void MonitorUserInterface::doSummary(void)
{
  if(!bei->addedContents.empty())
    checkAddedContents();
  
  for(vector<CollateMonitorElement *>::iterator it = collate_mes.begin(); 
      it != collate_mes.end(); ++it)
    {
      (*it)->summary();
    }
}

// this is the "main" loop where we receive monitoring;
// returns success flag
bool MonitorUserInterface::update(void)
{

  // retrieval of monitoring, sending of subscription requests/cancellations,
  // calculation of "collate"-type Monitoring Elements;
  bool ret = doMonitoring();

  // Run quality tests (and determine updated contents)
  runQTests();

  return ret;
}

