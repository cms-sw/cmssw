#include "DQMServices/Core/src/ClientServerRoot.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <TObjectTable.h>
#include <TDirectory.h>

#include <iostream>

using std::string;
using std::vector;
using std::cout; using std::endl;

using namespace dqm::monitor_data;

// ctor #1: configure client with collector hostname & port it should connect to
ClientServerRoot::ClientServerRoot(string host, unsigned port, 
				   string client_name, bool keepStaleSources,
				   bool sendLocalMEsOnly)
  : ClientRoot(host, port, client_name, keepStaleSources, true), 
    SenderBase(client_name, 0, sendLocalMEsOnly)
{
  init();
}

// ctor #2: client configuration for connection w/ multiple collectors
ClientServerRoot::ClientServerRoot(vector<string> hosts, unsigned port, 
				   string client_name, bool keepStaleSources,
				   bool sendLocalMEsOnly)
  : ClientRoot(hosts, port, client_name, keepStaleSources), 
    SenderBase(client_name, 0, sendLocalMEsOnly)
{
  init();
}

ClientServerRoot::~ClientServerRoot()
{

}

// to be called by ctors
void ClientServerRoot::init()
{
  s_fail_consec_max = 10;
  actAsServer_= true;  clients.clear();
  SenderBase::bei = DaqMonitorBEInterface::instance();
}

// remove node when disconnected;
void ClientServerRoot::removeSocket(int s, bool graceful_exit, 
				    bool add2collPending)
{
  if(isSource[s])
    {
      // have base classes do clean-up of monitoring elements associated w/ TSocket
      // first the sender side: loop over all clients
      for(ClientMap::iterator it = clients.begin(); it != clients.end(); ++it)
	{
	  currs = it->first;
	  setSenderPtrs();
	  SenderBase::cleanupSender(sources[s].name);
	}

      // then, the receiver side
      ClientRoot::removeSocket(s, graceful_exit, add2collPending);

      // need to send new monitorable to clients; maybe not the whole loop?
      doClientLoop();

    }
  else
    {
      // what cleanup do I need to do when a client goes down?
      removeClient(s);
    }

  showStats();
}

// come here to remove a client
void ClientServerRoot::removeClient(int s)
{
  currs = s;
  string name = clients[currs].name;
  cout << " Removing client " << name << " ..." << endl;

  setSenderPtrs();
  SenderBase::cleanupReceiver();

  // remove client from maps
  clients.erase(currs);
  removeSocket_base(name);
  ++closedClients;
}

// check for new source/client subscribing; return success flag
bool ClientServerRoot::newNode(string & buffer)
{
 if(newSource(buffer))
    return true;

 if(newClient(buffer))
   return true;

 // if here, the message needs to be processed in ReceiverBase or SenderBase; 
  // save the buffer
 if(isSource[currs])
   ReceiverBase::buffer_ = &buffer;
  else
    SenderBase::buffer_ = &buffer;
 
 return false;

}

// check if node has sent name & is identified as client; 
// add to client list if appropriate
bool ClientServerRoot::newClient(const std::string & buffer)
{
  if(buffer.find(nameClientPrefix) != string::npos)
    {
      // ------------------------------------------
      // Client has sent its name
      // ------------------------------------------
      TString name = buffer.c_str() + nameClientPrefix.size();
      string sname = name.Data();
      string hostname = getHostname(currs);
      cout << " Added client identified as " << sname << " at "
	   << hostname << endl;

      checkDroppedNodes(sname, hostname);
      insertName(sname);
      addReceiver(sname);

      clients[currs].Dir = SenderBase::makeDirStructure(sname);
      isSource[currs] = false;
      SenderBase::introduceYourself(currs, nameSourcePrefix);
      // need to send monitorable to new client; maybe not a loop over all clients?
      doClientLoop();
      return true;
    }

  return false;
}

// add TSocket to set of receivers/clients
void ClientServerRoot::addReceiver(string name)
{
  clients[currs] = ReceiverData();
  clients[currs].name = name;
}

// b4 calling sender: set addresses for socket, message & monitoring structure
void ClientServerRoot::setSenderPtrs()
{
  SenderBase::send_socket = currs;
  SenderBase::send_mess = mess; // used only for getting subscription
  SenderBase::receiver_ = &clients[currs];
}

// show statistics
void ClientServerRoot::showStats()
{
  if(ReceiverBase::bei->getVerbose() == 0) return;
  string longline=" ------------------------------------------------------------";
  cout << longline << endl;
  cout<<        "                          Update                             " 
      << endl << longline << endl;
  cout << " # of alive connections: " << getNumSourcesOn() << " sources and " 
       << getNumClientsOn() <<  " clients"
       << "\n # of closed connections: " << getNumSourcesOff() 
       << " (sources), " << getNumClientsOff() << " (clients) " << endl;
  cout << " # of monitoring updates: " << endl;
  for(SourceMap::iterator it = sources.begin(); it != sources.end(); ++it)
    cout << " Have received " << sources[it->first].count 
	 << " packages from source " << it->second.name << " at " 
	 << getHostname(it->first) << endl;
  for(ClientMap::iterator it = clients.begin(); it != clients.end(); ++it)
    cout << " Have sent " << clients[it->first].count << 
      " packages to client " << it->second.name << " at "
	 << getHostname(it->first) << endl;
  
  cout << " Total # of monitoring updates: received = " << getNumReceived() 
       << ", sent = " << getNumSent() << endl;
  cout << " # of monitoring packages received since last cycle: " << endl;
  for(SourceMap::iterator it = sources.begin(); it != sources.end(); ++it)
    cout << " Source " << it->second.name << ": " 
	 << it->second.cycles_count << endl;
  cout << " # of failed attempts to send monitoring: " << endl;
  bool foundAny = false;
  for(ClientMap::iterator it = clients.begin(); it != clients.end(); ++it)
    if(it->second.n_failed)
      {
	cout <<" Client " << it->second.name << ": " 
	     << it->second.n_failed << " times " << endl;
	foundAny = true;
      }
  if(!foundAny)cout << " (None) " << endl;
  cout << longline << endl;
}

// attempt to send monitoring, quality test results
void ClientServerRoot::doClientLoop()
{
  // do not send anything if we are in the middle of receiving
  if(!allReceiversDone())
    return;

  // true if monitoring objects have been sent to at least one client
  bool stuff_sent = false;
  SenderBase::startSending();
  try
    {
      // loop over all clients
      for(ClientMap::iterator it = clients.begin(); it != clients.end(); 
	  ++it)
	{
	  currs = it->first;
	  setSenderPtrs();
	  int N = SenderBase::send();
	  if(N > 0)
	    stuff_sent = true;
	  if(N < 0)
	    checkClient();
	  
	} // loop over all clients
    }
  catch (bool e)
    {
      if(!e) cout << " Exception caught in SenderBase::send method" 
		  << endl;
    }
  catch (...)
    {
      cout << " *** Error! Unexpected exception caught in doClientLoop!" << endl;
    }

  if(stuff_sent)
    {
      // loop over all sources, reset counter of monitoring cycles
      for(SourceMap::iterator it = sources.begin(); it != sources.end(); 
	  ++it)
	it->second.cycles_count = 0;
    }

  SenderBase::doneSending(shouldResetMEs, shouldCallResetDiff);
}

// check whether all receivers are done sending
bool ClientServerRoot::allReceiversDone()
{
  // loop over all clients
  for(ClientMap::iterator it = clients.begin(); it != clients.end(); 
      ++it)
    {
      currs = it->first;
      setSenderPtrs();

      if(!SenderBase::isReceiverDone())
	return false;
    } // loop over all clients

  // if here, all clientes are done sending
  return true;
}


// check whether all senders are done sending
bool ClientServerRoot::allSendersDone()
{
  // FOR NOW: check if all senders are done sending monitorable and monitoring
  for(SourceMap::iterator it = sources.begin(); it != sources.end(); ++it)
    {
      currs = it->first;
      setReceiverPtrs();
      if(!ReceiverBase::isSenderDone())
	return false;
    } // loop over all sources

  // if here, all sources are done sending
  return true;
}

// come here to determine if client has gone down
void ClientServerRoot::checkClient()
{
  if(clients[currs].n_failed)
    {
      if(clients[currs].n_failed_consec % s_fail_consec_max == 0)
	{
	  cout << " *** Failed to communicate with client " 
	       << clients[currs].name << " " 
	       << clients[currs].n_failed_consec << " consecutive times"
	       << endl;
	  cout << " Removing client... " << endl;
	  removeClient(currs);
	}
    }
}

// get # of names for connected sources
vector<string> ClientServerRoot::getSourceNames() const
{
  vector<string> ret;
  for(SourceMap::const_iterator it = sources.begin(); it != sources.end(); 
      ++it)
    ret.push_back(it->second.name);
  return ret;
}

// get # of names for connected clients
vector<string> ClientServerRoot::getClientNames() const
{
  vector<string> ret;
  for(ClientMap::const_iterator it = clients.begin(); it != clients.end(); 
      ++it)
    ret.push_back(it->second.name);
  return ret;
}

// get host name for node (source or client)
string ClientServerRoot::getHostName(const string & node_name) const
{
  NameSocketMap::const_iterator it = all_names.find(node_name);
  if(it == all_names.end())
    {
      cout << " *** Unknown node name = " << node_name << " in getHostName!"
	   << endl;
      return "0.0.0";
    }

  return getHostname(it->second); 
}

// get # of monitoring packages received
unsigned ClientServerRoot::getNumReceived(const string & source_name) const
{
  NameSocketMap::const_iterator it = all_names.find(source_name);
  if(it == all_names.end())
    {
      cout << " *** Unknown node name = " << source_name 
	   << " in getNumReceived!" << endl;
      return 0;
    }

  SocketBoolMap::const_iterator bit = isSource.find(it->second);
  if(!bit->second)
    {
      cout << " *** getNumReceived cannot be used for client " << source_name
	   << endl;
      return 0;
    }

  SourceMap::const_iterator sit = sources.find(it->second);
  return sit->second.count;

}

// get # of monitoring packages sent
unsigned ClientServerRoot::getNumSent(const string & client_name) const
{
  NameSocketMap::const_iterator it = all_names.find(client_name);
  if(it == all_names.end())
    {
      cout << " *** Unknown node name = " << client_name << " in getNumSent!"
	   << endl;
      return 0;
    }

  SocketBoolMap::const_iterator bit = isSource.find(it->second);
  if(bit->second)
    {
      cout << " *** getNumSent cannot be used for source " << client_name
	   << endl;
      return 0;
    }

  ClientMap::const_iterator cit = clients.find(it->second);
  return cit->second.count;

}

// returns true if full monitoring set (or command) has been received
bool ClientServerRoot::unpackData()
{
  if(isSource[currs])
    {
      // receive monitoring from source
      return ClientRoot::unpackData();
    }
  else
    {
      // receive subscription request from client
      setSenderPtrs();
      SenderBase::getSubscription();      
      // return control only when we have received full subscription request
      return SenderBase::isReceiverDone();
    }
}

void ClientServerRoot::summary(bool checkMemoryLeaks)
{
  if(SenderBase::bei->getVerbose() > 0)showStats();
  if(checkMemoryLeaks)
    {
      checkMemory();
      gDirectory->GetList()->Print();
      gObjectTable->Print();
      ReceiverBase::saveFile("test.root");
    }
}

// attempt to send monitoring, quality test results
void ClientServerRoot::sendStuff()
{
  // **** Note: *****
  // ignore client's requests till all senders/sources are done sending may be
  // a little too strict if (a) too many sources (b) update rate too high
  if(allSendersDone())
    // here we attempt to send monitorable/monitoring/test results to clients
    doClientLoop();
}
