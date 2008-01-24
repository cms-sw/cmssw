#include "DQMServices/Core/interface/MonitorElement.h"
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
  myc = 0;
  standaloneMode_ = true;
  init();
}

MonitorUserInterface::~MonitorUserInterface(void)
{
  if(myc)
    { delete myc; myc = 0; }

}

// to be used by methods subscribe (add=true) and unsubscribe (add=false)
// <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
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
void MonitorUserInterface::subscribe(string subsc_request)
{
  subscribe_base(subsc_request, true, bei->Own);
}

// unsubscription request; read description in similar subscription method
void MonitorUserInterface::unsubscribe(string subsc_request)
{
  subscribe_base(subsc_request, false, bei->Own);
}

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

// retrieval of monitoring, sending of subscription requests/cancellations,
// returns success flag
bool MonitorUserInterface::doMonitoring(void)
{
  // initialization needed at beginning of monitoring cycle
  bei->resetMonitorableDiff();
  bei->resetMonitoringDiff();

  // there is no receiving to be done when there are 
  // no upstream and no downstream connections
  if(!needUpstreamConnections() && !isServer())
    return false;

  if(needUpstreamConnections() && myc->noUpstreamConnections())
    {
      cout << " *** No connections with collectors are possible..." << endl;
      return false;
    }

  bool ret = myc->update();
  int num = myc->getNumUpdates();
  
  bool newStuff = false;
  if(num != numUpdates_)
    {
      numUpdates_ = num;
      newStuff = true;
    }

  return ret;
}

// allow downstream clients to connect to this client
// (to be used only with no-arg ctor; use boolean switch for other ctors)
void MonitorUserInterface::actAsServer(int port_no, std::string client_name)
{
  if(!myc)
    {
      myc = new ClientServerRoot("", port_no, client_name, false, true);
      init();
    }
}

// return # of monitoring cycles received
int MonitorUserInterface::getNumUpdates(void) const
{
  if(needUpstreamConnections()) return myc->getNumUpdates();
  else return -1;
}

bool MonitorUserInterface::isConnected(void) 
{
  if (standaloneMode_) return false ; 
  return myc->isConnected() ;
} 

// like subscribe_base in base class, for one path only
void MonitorUserInterface::subscribe_base(const string & subsc_request, bool add,
				   vector<string> & requests, 
				   const MonitorElementRootFolder * folder)
{
  string new_request;
  std::string pathname = folder->getPathname();
  for(cME_it it = folder->objects_.begin(); it != folder->objects_.end(); 
      ++it)
    { // loop over files in folder
	  
      // "subscribe" should loop over "monitorable";
      // "unsubscribe" should loop over "contents" (only non-null MEs)
      
      // if unsubscribing, skip null MEs
      if(!add && !it->second)continue;
      // if subscribing, skip non-null MEs
      if(add && it->second)continue;

      string fullname = getUnixName(pathname, it->first);
	  
      if(matchit(fullname, subsc_request))
	{ // this is a match!
	  
	  if(!new_request.empty())new_request += ",";
	  new_request += it->first;
	  
	} // this is a match!
	  
    }  // loop over files in directory <path->first>
      
  if(!new_request.empty())
    requests.push_back(pathname + ":" + new_request);
  
}
// (un)subscription request for directory contents ==> FAST
// (need exact pathname without wildcards, e.g. A/B/C);
// use flag to specify whether subfolders (and their contents) should be included;
// use add=true(false) to (un)subscribe

/*void MonitorUserInterface::subscribeDir(string & subsc_request, bool useSubFolders,
				 unsigned int myTag, bool add)
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

  MonitorElementRootFolder * folder = 0;
  if(myTag)
    {
      ctdir_it tg = bei->Tags.find(myTag);
      if(tg == bei->Tags.end())
	cout << " *** Tag " << myTag << " does not exist! " << endl;
      else
	folder = bei->getDirectory(subsc_request, tg->second);
    }
  else
    folder = bei->getDirectory(subsc_request);

  if(!folder)
    {
      cerr << " *** Directory " << subsc_request << " does not exist!\n";
      cout << " Subscription request failed... " << endl;
      return;
    }

  subscribeDir(folder, useSubFolders, myTag, add);
}
*/

// same as above for MonitorElementRootFolder
void MonitorUserInterface::subscribeDir(MonitorElementRootFolder * folder, 
				 bool useSubFolders,
				 unsigned int myTag, bool add)
{
  assert(folder);

  // put here requests in format <pathname:> or <pathname::myTag>
  vector<string> requests;
  
  if(useSubFolders)
    { // use all subfolders
      typedef vector<MonitorElementRootFolder *> vMERF; vMERF all_subfolds;
      folder->getSubfolders(all_subfolds);
      for(vMERF::const_iterator subf = all_subfolds.begin(); 
	  subf != all_subfolds.end(); ++subf)
	{ // loop over all subfolders
	  
	  if(! ((*subf)->objects_.empty()) )
	    // directory has at least one child: send subscription request
	    addFolderSubsc(*subf, myTag, requests);
	  
	} // loop over all subfolders

    } // use all subfolders
  
  if(! folder->objects_.empty())
    addFolderSubsc(folder, myTag, requests);

  // (un)subscribe only if non-zero list has been found
  if(!requests.empty())
    finishSubscription(requests, add);
}

// add <pathname:> or <pathname::myTag> (if myTag != 0) to requests
void MonitorUserInterface::addFolderSubsc(MonitorElementRootFolder * folder,
				   unsigned int myTag, 
				   vector<string> & requests) const
{
  string subscription = folder->getPathname() + ":";
  if(myTag) 
    {
      std::ostringstream tagName; tagName << myTag;
      subscription += ":" + tagName.str();
    }
  requests.push_back(subscription);
}

// to be called by ctors
void MonitorUserInterface::init()
{  
  bei = DaqMonitorBEInterface::instance();

  if(myc)
    {
      // see comment in ClientRoot::shouldResetMEs
      myc->setShouldResetMEs(!needUpstreamConnections()); 
      // see comment in ClientRoot::setShouldCallResetDiff
      myc->setShouldCallResetDiff(false);

      // call to update will NOT block since we run in standalone mode
      if(!needUpstreamConnections())
	myc->doNotBlockUpdate = true;
    }      
}
// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
// using <client_name>; if flag=true, will accept downstream connections
MonitorUserInterface::MonitorUserInterface(string hostname, int port_no, string client_name, 
			     int reconnect_delay_secs, bool actAsServer) 
{
  standaloneMode_ = false;

  bool keepStaleSources = false;
  try{
    if(actAsServer)
      {
	bool sendLocalMEsOnly = true;
	myc = new ClientServerRoot(hostname, port_no, client_name, 
				   keepStaleSources, sendLocalMEsOnly);
	myc->setReconnectDelay(reconnect_delay_secs);
      }
    else
      {
	bool multipleConnections = false;
	myc = new ClientRoot(hostname, port_no, client_name,
			     keepStaleSources, multipleConnections);
	myc->setReconnectDelay(reconnect_delay_secs);
      }
  }
  catch (ClientRoot * cr)
    {
      if(!cr) throw (MonitorUserInterface * ) 0;
    }

  init();
}

// Connect with monitoring server (DQM Collector) with a list of hostnames at 
// <port_no> using <client_name>; if flag=true, will accept downstream connections
MonitorUserInterface::MonitorUserInterface(vector<string> hostnames, int port_no, 
			     string client_name, int reconnect_delay_secs,
			     bool actAsServer) 
{
  standaloneMode_ = false;

  bool keepStaleSources = false;
  try{
    if(actAsServer)
      {
	bool sendLocalMEsOnly = true;
	myc = new ClientServerRoot(hostnames, port_no,client_name, 
				   keepStaleSources, sendLocalMEsOnly);
	myc->setReconnectDelay(reconnect_delay_secs);
      }
    else
      myc = new ClientRoot(hostnames, port_no, client_name,
			   keepStaleSources);
	myc->setReconnectDelay(reconnect_delay_secs);
  }
  catch (ClientRoot * cr)
    {
      if(!cr) throw (MonitorUserInterface * ) 0;
    }

  init();
}

