#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/src/ClientRoot.h"
#include "DQMServices/Core/interface/MonitorUIRoot.h"

#include "DQMServices/Core/interface/NodeBase.h"
#include "DQMServices/Core/interface/MonitorData.h"

using namespace dqm::me_util;

using std::cout; using std::cerr; using std::endl;
using std::string; using std::vector;

// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
// using <client_name>; if flag=true, will accept downstream connections
MonitorUIRoot::MonitorUIRoot(string hostname, int port_no, string client_name, 
			     int reconnect_delay_secs, bool actAsServer) : 
  MonitorUserInterface() 
{
  standaloneMode_ = false;

  bool keepStaleSources = false;
  try{
    if(actAsServer)
      {
	bool sendLocalMEsOnly = true;
	myc = new ClientServerRoot(hostname, port_no, client_name, 
				   keepStaleSources, sendLocalMEsOnly);
      }
    else
      {
	bool multipleConnections = false;
	myc = new ClientRoot(hostname, port_no, client_name,
			     keepStaleSources, multipleConnections);
      }
  }
  catch (ClientRoot * cr)
    {
      if(!cr) throw (MonitorUIRoot * ) 0;
    }

  setReconnectDelay(reconnect_delay_secs);
  init();
}

// Connect with monitoring server (DQM Collector) with a list of hostnames at 
// <port_no> using <client_name>; if flag=true, will accept downstream connections
MonitorUIRoot::MonitorUIRoot(vector<string> hostnames, int port_no, 
			     string client_name, int reconnect_delay_secs,
			     bool actAsServer) : 
  MonitorUserInterface() 
{
  standaloneMode_ = false;

  bool keepStaleSources = false;
  try{
    if(actAsServer)
      {
	bool sendLocalMEsOnly = true;
	myc = new ClientServerRoot(hostnames, port_no,client_name, 
				   keepStaleSources, sendLocalMEsOnly);
      }
    else
      myc = new ClientRoot(hostnames, port_no, client_name,
			   keepStaleSources);
  }
  catch (ClientRoot * cr)
    {
      if(!cr) throw (MonitorUIRoot * ) 0;
    }

  setReconnectDelay(reconnect_delay_secs);

  init();
}

// Use the default constructor for running in standalone mode 
// (ie. without sources or collectors);
MonitorUIRoot::MonitorUIRoot() : MonitorUserInterface()
{
  myc = 0;
  standaloneMode_ = true;
  init();
}

// to be called by ctors
void MonitorUIRoot::init()
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

// destructor
MonitorUIRoot::~MonitorUIRoot(void)
{
  if(myc)
    { delete myc; myc = 0; }
}

// retrieval of monitoring, sending of subscription requests/cancellations,
// returns success flag
bool MonitorUIRoot::doMonitoring(void)
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

// return # of monitoring cycles received
int MonitorUIRoot::getNumUpdates(void) const
{
  if(needUpstreamConnections())
    return myc->getNumUpdates();
  else
    return -1;
}

// Run quality tests on all MonitorElements that have been updated (or added)
// since last monitoring cycle;
// send monitoring, results downstream if applicable
void MonitorUIRoot::runQTests(void)
{
   bei->runQTests();
   if(myc) myc->sendStuff(); 
}

// true if name appear as child of directory (even if it is null MonitorElement)
bool MonitorUIRoot::objectDefined(const string & name, const string & pathname) 
  const
{
  MonitorElementRootFolder * folder = bei->getDirectory(pathname);
  if(!folder)return false;
  bool ret = (folder->objects_.find(name) != folder->objects_.end());    
  if(ret)
    {
      cout << " *** Attempt to redefine MonitorElement " << name 
	   << " in " << pathname << endl;
    }
  return ret;
}

// allow downstream clients to connect to this client
// (to be used only with no-arg ctor; use boolean switch for other ctors)
void MonitorUIRoot::actAsServer(int port_no, std::string client_name)
{
  if(!myc)
    {
      myc = new ClientServerRoot("", port_no, client_name, false, true);
      init();
    }
}

// add call back; to be used for thread-unsafe operations
void MonitorUIRoot::addCallback(seal::Callback & action)
{
  if(myc)
    {
      myc->addCallback(action);
    }
}

// like subscribe_base in base class, for one path only
void MonitorUIRoot::subscribe_base(const string & subsc_request, bool add,
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
void MonitorUIRoot::subscribeDir(string & subsc_request, bool useSubFolders,
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

// same as above for MonitorElementRootFolder
void MonitorUIRoot::subscribeDir(MonitorElementRootFolder * folder, 
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
void MonitorUIRoot::addFolderSubsc(MonitorElementRootFolder * folder,
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
