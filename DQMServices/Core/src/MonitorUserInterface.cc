#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include <iostream>

using namespace dqm::me_util;
using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector;

/* Use the default constructor for running in standalone mode (ie. without
   sources or collectors); if flag=true, client will accept downstream connections
*/
MonitorUserInterface::MonitorUserInterface()
{
  numUpdates_ = 0; bei = 0;
}

MonitorUserInterface::~MonitorUserInterface(void)
{
  bei->removeCollates();
}

// to be used by methods subscribe (add=true) and unsubscribe (add=false)
// <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
void MonitorUserInterface::subscribe_base(const string & subsc_request, bool add,
					  const rootDir & Dir)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  if(!hasWildCards(subsc_request))
    { // exact pathname here
      MonitorElement * me = bei->get(subsc_request);
      // if unsubscribing, skip null MEs
      if(!add && !me)return;
      // if subscribing, skip non-null MEs
      if(add && me)return;

      string path, filename;
      unpack(subsc_request, path, filename); 	 
      requests.push_back(path + ":" + filename);
    }
  else
    {
      cdir_it start, end, parent_dir;
      getSubRange<dir_map>(subsc_request, Dir.paths, 
			   start, end, parent_dir);
      
      for(cdir_it path = start; path != end; ++path)
	// loop over pathnames in directory structure
	subscribe_base(subsc_request, add, requests, path->second);
      
      if(parent_dir != Dir.paths.end())
	subscribe_base(subsc_request, add, requests, parent_dir->second);
    }

  // (un)subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, add);
  
}

// subscription request; format: (a) exact pathname (e.g. A/B/C/histo) ==> FAST
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*) ==> SLOW
void MonitorUserInterface::subscribe(string subsc_request)
{
  subscribe_base(subsc_request, true, bei->Own);
}

// unsubscription request; read description in similar subscription method
void MonitorUserInterface::unsubscribe(string subsc_request)
{
  subscribe_base(subsc_request, false, bei->Own);
}

// same as above for tagged MonitorElements
void MonitorUserInterface::subscribe(string subsc_request, unsigned int tag)
{
  ctdir_it tg = bei->Tags.find(tag);
  if(tg == bei->Tags.end())
    return;
  subscribe_base(subsc_request, true, tg->second);
}

// unsubscription request; read description in similar subscription method
void MonitorUserInterface::unsubscribe(string subsc_request, unsigned int tag)
{
  ctdir_it tg = bei->Tags.find(tag);
  if(tg == bei->Tags.end())
    return;
  subscribe_base(subsc_request, false, tg->second);
}

// subscription request for directory contents ==> FAST
// (need exact pathname without wildcards, e.g. A/B/C);
// use flag to specify whether subfolders (and their contents) should be included;
// Users are encourage to use this method instead of previous one w/ wildcards
void MonitorUserInterface::subscribe(string subsc_request, bool useSubFolders)
{
  subscribeDir(subsc_request, useSubFolders, 0, true);
}

// unsubscription request; read description in similar subscription method
void MonitorUserInterface::unsubscribe(string subsc_request, bool useSubFolders)
{
  subscribeDir(subsc_request, useSubFolders, 0, false);
}

// same as above for tagged MonitorElements
void MonitorUserInterface::subscribe(string subsc_request, bool useSubFolders,
				     unsigned int tag)
{
  subscribeDir(subsc_request, useSubFolders, tag, true);
}

// unsubscription request; read description in similar subscription method
void MonitorUserInterface::unsubscribe(string subsc_request, bool useSubFolders,
				       unsigned int tag)
{
  subscribeDir(subsc_request, useSubFolders, tag, false);
}

// subscription request for all MEs with given tag ==> FAST
void MonitorUserInterface::subscribe(unsigned int tag)
{
  ctdir_it tg = bei->Tags.find(tag);
  if(tg == bei->Tags.end())
    cout << " *** Tag " << tag << " does not exist! " << endl;
  else
    {
      bool useSubFolders = true; 
      subscribeDir(tg->second.top, useSubFolders, tag, true);
    }
}

// unsubscription request for all MEs with given tag ==> FAST
void MonitorUserInterface::unsubscribe(unsigned int tag)
{
  ctdir_it tg = bei->Tags.find(tag);
  if(tg == bei->Tags.end())
    cout << " *** Tag " << tag << " does not exist! " << endl;
  else
    {
      bool useSubFolders = true; 
      subscribeDir(tg->second.top, useSubFolders, tag, false);
    }
}


// use only additions to monitorable in last cycle; subscription request format: 
// (a) exact pathname with ME name (e.g. A/B/C/histo) ==> FAST
// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*) ==> SLOW
void MonitorUserInterface::subscribeNew(string subsc_request)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  vector<string> put_here; put_here.clear();
  bei->getAddedMonitorable(put_here);
  
  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  DirFormat dir;
  for(cvIt mon = put_here.begin(); mon != put_here.end(); ++mon)
    { // loop over added monitorable's pathnames
      if(!unpackDirFormat(*mon, dir))
	continue;
      
      if(!hasWildCards(subsc_request))
	{
	  string subsc_path, subsc_filename;
	  unpack(subsc_request, subsc_path, subsc_filename);
	  if(dir.dir_path != subsc_path)continue;
	}
      
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

// same as above for tagged MonitorElements (with modification in last cycle only)
void MonitorUserInterface::subscribeNew(string subsc_request, unsigned int tag)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  // will consider only addedTags (as opposed to addedMonitorable)

  if(!hasWildCards(subsc_request) )
    { // subscription request with exact pathname
      string subsc_path, subsc_filename;
      unpack(subsc_request, subsc_path, subsc_filename);
      
      cdirt_it path = bei->addedTags.find(subsc_path);
      if(path != bei->addedTags.end())
	{ // found path in addedTags
	  ctags_it me = path->second.find(subsc_filename);
	  if(me != path->second.end())
	    { // found ME in addedTags
	      
	      if(me->second.find(tag) != me->second.end())
		requests.push_back(subsc_path + ":" + subsc_filename);
	      
	    } // found ME in addedTags
	} // found path in addedTags
      
    } // subscription request with exact pathname
  
  else // subscription request with wildcards
    {
      
      cdirt_it start, end, parent_dir;
      getSubRange<dir_tags>(subsc_request, bei->addedTags, 
			    start, end, parent_dir);
      
      for(cdirt_it path = start; path != end; ++path)
	// loop over pathnames in addedTags
	subscribeNew(subsc_request, tag, requests, path);
      
      if(parent_dir != bei->addedTags.end())
	subscribeNew(subsc_request, tag, requests, parent_dir);
      
    } // subscription request with wildcards


  // subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, true);
  
}

// look for MEs matching subsc_request with <tag> in <path>
void MonitorUserInterface::subscribeNew(const string & subsc_request,
					unsigned int tag,
					vector<string> & requests,
					const cdirt_it & path)
{
  string new_request;

  for(ctags_it me = path->second.begin();  me != path->second.end(); ++me)
    { // loop over MEs in <path>


      // check if tag has been added to ME
      if(me->second.find(tag) != me->second.end())
	{
		     
	  string fullname = getUnixName(path->first, me->first);
		     
	  if(matchit(fullname, subsc_request))
	    { // this is a match!

	      if(!new_request.empty())new_request += ",";
	      new_request += me->first;
			 
	    } // this is a match!
	  
	} // check if tag has been added to ME

    } // loop over all MEs in <path>

  if(!new_request.empty())
    requests.push_back(path->first + ":" + new_request);

}

// get all MEs with <tag> in <path>
void MonitorUserInterface::subscribeNew(unsigned int tag, 
					vector<string> & requests,
					const cdirt_it & path)
{
  string new_request;

  for(ctags_it me = path->second.begin();  me != path->second.end(); ++me)
    { // loop over MEs in <path>


      // check if tag has been added to ME
      if(me->second.find(tag) != me->second.end())
	{
		     
	  if(!new_request.empty())new_request += ",";
	  new_request += me->first;

	} // check if tag has been added to ME

    } // loop over all MEs in <path>

  if(!new_request.empty())
    requests.push_back(path->first + ":" + new_request);
  
}

// subscription request for directory contents ==> FAST
// use only additions to monitorable in last cycle;
// (need exact pathname without wildcards, e.g. A/B/C)
// use flag to specify whether subfolders (and their contents) should be included;
void MonitorUserInterface::subscribeNew(string subsc_request, bool useSubFolders)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  if(hasWildCards(subsc_request))
    {
      cerr << " *** Wildcards (*, ?) not allowed with this subscription method"
	   << endl;
      return;
    }

  chopLastSlash(subsc_request);

  vector<string> put_here; put_here.clear();
  bei->getAddedMonitorable(put_here);
  
  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  DirFormat dir;
  for(cvIt mon = put_here.begin(); mon != put_here.end(); ++mon)
    { // loop over added monitorable's pathnames

      if(useSubFolders)
	{ // need all subfolders;
	  // will check if (*mon) contains (folder) subc_request
	  if(!belongs2folder(subsc_request, *mon))
	    continue;
	}
      else // no subfolders needed;
	{ // will check for exact pathname

	  // split name into <dir> and contents
	  vector<string> subs; 
	  unpackString((*mon).c_str(), ":", subs);
	  if(subs.size() != 2)
	    {
	      cout << " *** Error parsing added monitorable " 
		   << *mon << endl;
	      continue;
	    }

	  string path = *(subs.begin());
	  if(subsc_request != path)continue;
	}

      requests.push_back(*mon);
      
    } // loop over added monitorable's pathnames
  
  // subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, true);

}

// same as above for tagged MonitorElements (with modification in last cycle only)
void MonitorUserInterface::subscribeNew(string subsc_request, bool useSubFolders,
					unsigned int tag)
{
  if(!needUpstreamConnections()) return;
  if(subsc_request.empty())return;

  if(hasWildCards(subsc_request))
    {
      cerr << " *** Wildcards (*, ?) not allowed with this subscription method"
	   << endl;
      return;
    }

  chopLastSlash(subsc_request);

  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  if(useSubFolders)
    {
      // get all subfolders by using exact-pathname + "/*"
      string search_string = subsc_request + "/*";
      
      cdirt_it start, end, parent_dir;
      getSubRange<dir_tags>(search_string, bei->addedTags, 
			    start, end, parent_dir);    
 
      for(cdirt_it path = start; path != end; ++path)
	// loop over pathnames in addedTags
	subscribeNew(tag, requests, path);
      
      if(parent_dir != bei->addedTags.end())
	subscribeNew(tag, requests, parent_dir);
      
    } // subscription request with wildcards

  else
    { // use only <subsc_request> directory (no subfolders)
      cdirt_it path = bei->addedTags.find(subsc_request);
      if(path != bei->addedTags.end())
	subscribeNew(tag, requests, path);

    }
  
  // subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, true);

}


// subscription request for all MEs with given tag in last monitoring cycle==> FAST
void MonitorUserInterface::subscribeNew(unsigned int tag)
{
  if(!needUpstreamConnections()) return;

  // put here requests in format <dir pathname>:<h1>,<h2>,...
  vector<string> requests;

  for(cdirt_it path = bei->addedTags.begin(); path != bei->addedTags.end();
      ++path)
    // loop over pathnames in addedTags
    subscribeNew(tag, requests, path);
      
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

  for(cvIt it = monit.begin(); it != monit.end(); ++it)
    {
      
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
//void MonitorUserInterface::save(string filename, string dir_fullpath,
//				int minimum_status) const
//{
//  bei->save(filename, dir_fullpath, minimum_status);
//}

// draw all monitoring objects
//void MonitorUserInterface::drawAll(void) const
//{
//  bei->drawAll();
//}

// reset ME contents (does not permanently reset contents)
//void MonitorUserInterface::softReset(MonitorElement * me)
//{
//  bei->softReset(me);
//}

// reverts action of softReset
//void MonitorUserInterface::disableSoftReset(MonitorElement * me)
//{
//  bei->disableSoftReset(me);
//}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
//void MonitorUserInterface::setAccumulate(MonitorElement * me, bool flag)
//{
//  bei->setAccumulate(me, flag);
//}

// add <search_string> to summary ME; 
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
// this action applies to all MEs already available or future ones
void MonitorUserInterface::add(CollateMonitorElement * cme, string search_string)
  const
{
  add(cme, 0, search_string); // "0" means no tag
}

// same as above for tagged MEs
void MonitorUserInterface::add(CollateMonitorElement * cme, unsigned int tag,
			       string search_string) const
{
  if(search_string.empty())
    return;
  if(!cme)
    {
      cerr << " *** Cannot use search-string " << search_string 
	   << " with null CollateMonitorElement! " << endl;
      return;
    }

  if(tag == 0) // "0" means no tag
    cme->add(0, search_string, bei->Own);
  else
    {
      tdir_it tg = bei->Tags.find(tag);
      if(tg != bei->Tags.end())
	cme->add(tag, search_string, tg->second);
      else
	cme->add2search_path(search_string, tag);
    }
   
}

// add directory contents to summary ME ==> FAST
// (need exact pathname without wildcards, e.g. A/B/C);
// use flag to specify whether subfolders (and their contents) should be included;
// this action applies to all MEs already available or future ones
void MonitorUserInterface::add(CollateMonitorElement* cme, string pathname, 
			       bool useSubfolds) const
{
  add(cme, 0, pathname, useSubfolds); // "0" means no tag
}

// same as above for tagged MEs
void MonitorUserInterface::add(CollateMonitorElement * cme, unsigned int tag,
			       string pathname, bool useSubfolds) const
{
  if(pathname.empty())
    return;
  if(!cme)
    {
      cerr << " *** Cannot use pathname " << pathname 
	   << " with null CollateMonitorElement! " << endl;
      return;
    }

  chopLastSlash(pathname);

  if(tag == 0) // "0" means no tag
    cme->add(0, pathname, bei->Own, useSubfolds);
  else
    {
      tdir_it tg = bei->Tags.find(tag);
      if(tg != bei->Tags.end())
	cme->add(tag, pathname, tg->second, useSubfolds);
      else
	cme->add2folders(pathname, useSubfolds, tag);
    } 
}

// add tagged MEs to summary ME ==> FAST
// this action applies to all MEs already available or future ones
void MonitorUserInterface::add(CollateMonitorElement * cme, unsigned int tag)
  const
{
  if(!cme)
    {
      cerr << " *** Cannot use tag " << tag 
	   << " with null CollateMonitorElement! " << endl;
      return;
    }
  if(tag == 0)
    {
      cerr << " *** Tag must be positive number! \n";
      return;
    }

  tdir_it tg = bei->Tags.find(tag);
  if(tg != bei->Tags.end())
    cme->add(tag, tg->second);
  else
    cme->add2tags(tag); 
}

// new MEs have been added; check if need to update collate-MEs
void MonitorUserInterface::checkAddedContents(void)
{
  for(cmesIt cme = bei->collate_set.begin(); cme != bei->collate_set.end(); ++cme)
    // loop over collate-MEs
    (*cme)->checkAddedContents();
}

// do calculations for all collate MEs; come here at end of monitoring cycle)
void MonitorUserInterface::doSummary(void)
{
  if(!bei->addedContents.empty())
    checkAddedContents();
  
  for(cmesIt cme = bei->collate_set.begin(); cme != bei->collate_set.end(); ++cme)
    (*cme)->summary();
}

// this is the "main" loop where we receive monitoring/send subscription requests;
// if client acts as server, method runQTests is also sending monitoring & 
// test results to clients downstream;
// returns success flag
bool MonitorUserInterface::update(void)
{

  cout << " In MonitorUserInterface::update: " ;
  cout << " This method will be deprecated soon, please replace mui->update() by: " << endl; 
  cout << "     bool ret = mui->doMonitoring(); " << endl;
  cout << "     bei->runQTests(); " << endl;

  // retrieval of monitoring, sending of subscription requests/cancellations,
  // calculation of "collate"-type Monitoring Elements;
  bool ret = doMonitoring();

  // Run quality tests (and determine updated contents);
  // Method is overloaded if client acts as server to other clients downstream
  bei->runQTests();

  return ret;
}

