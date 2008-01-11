#include "DQMServices/Core/interface/MonitorData.h"
#include "DQMServices/Core/src/ClientRoot.h"
#include "DQMServices/Core/interface/SocketUtils.h"
#include "DQMServices/Core/interface/DQMMessage.h"

#include "TObjString.h"
#include "TObjectTable.h"
#include "TDirectory.h"
#include "TROOT.h"

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>
#include <arpa/inet.h>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::map;

using namespace dqm::me_util;
using namespace dqm::monitor_data;

int ClientRoot::ss = -1;

// ctor #1: configure client with collector hostname & port it should connect to
ClientRoot::ClientRoot(string host, unsigned port, string client_name, 
		       bool keepStaleSources, bool multipleConnections) : 
  ReceiverBase(nameClientPrefix + client_name, keepStaleSources), 
  multipleConnections_(multipleConnections),
  monMax (-1), selectCalled (false)

{
  mess = new DQMMessage;
  init(port, defTimeoutMsecs);
  connect2Server(host);
}

// ctor #2: client configuration for connection w/ multiple collectors
ClientRoot::ClientRoot(vector<string> hosts, unsigned port, string client_name,
		       bool keepStaleSources) :
  ReceiverBase(nameClientPrefix + client_name, keepStaleSources),
  multipleConnections_(true), monMax (-1), selectCalled (false)
{
  mess = new DQMMessage;
  init(port, defTimeoutMsecs);
  for(cvIt it = hosts.begin(); it != hosts.end(); ++it)
    connect2Server(*it);
}

ClientRoot::~ClientRoot()
{
  disconnect();
}

// initialization done at startup
void ClientRoot::init(unsigned port, int timeout_msecs)
{
  actAsServer_ = false; currs = -1; //ss = -1;
  maxAttempts2Reconnect = MAX_RECON;

  if(multipleConnections())
    {
      // Open a server socket looking for connections 
      //  on a named service or on a specified port
      ss = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
      sockaddr_in *s_address = new sockaddr_in ();
      s_address->sin_addr.s_addr = INADDR_ANY;
      s_address->sin_port = htons (port);
      s_address->sin_family=AF_INET;
      int  val = 1;
      setsockopt(ss, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
      //used to avoid problems with unbinding
      int status = bind (ss, (sockaddr*) s_address, sizeof (sockaddr_in));
      listen (ss,5);
      monMax = ss;
      if (status < 0)
	{
	  int e = errno;
	  char ee[256];
	  puts(strerror_r(e,ee,256));
	  cout << " *** Server already running at port " << port << endl;
	  cout << " (Can only run one collector or client on same machine) " 
	       << endl;
	  cout << " Exiting... " << endl;
	  throw (ClientRoot *) 0;
 	}
    }
  serv_hostnames.clear(); isSource.clear(); sources.clear(); 
  droppedNodes.clear(); all_names.clear(); closedSources = 0;
  updates_ = 0;
  setPort(port);
  setReconnectDelay(5);
  setWait2Subscribe(timeout_msecs);
  // get hold of back-end interface instance
  ReceiverBase::bei = DaqMonitorROOTBackEnd::instance();
  
  // Should be set to true by MonitorUIRoot if client runs in standalone mode
  doNotBlockUpdate = false;
  
  // initial values; they should be set by inheriting class
  shouldResetMEs = false;
}

// attempt to connect to hostname at port #;
// Use isConnected to test if connection has been made
void ClientRoot::connect(string host, unsigned port)
{
  setPort(port);  
  connect2Server(host);
}

// come here to attempt connection with collector at <host>
void ClientRoot::connect2Server(const string & host)
{
  // protection against null hostnames
  if(host == "")return;

  if(serv_hostnames.find(host) != serv_hostnames.end())
    {
      cout << " *** Already connected to " << host 
	   << "! Connection request ignored..." << endl;
      return;
    }
  //
  setHost(host);

  int sock = NodeBase::connect ("");
  // attempt connection
  if(sock >= 0)
    {
      // better to use getHostname (e.g. lxplus019.cern.ch) instead of host
      // (could be e.g. just lxplus019)
      serv_hostnames[getHostname(sock)] = sock;
      addSocket(sock);
      if(collectorsPending.find(host) != collectorsPending.end())
	{
	  collectorsPending.erase(host);
	}
    }
  else
    {
      // connection failed, will need to retry
      collectorsPending[host].when = TTimeStamp();
      collectorsPending[host].attempts++;
      int N = maxAttempts2Reconnect - collectorsPending[host].attempts;
      if(N)
	{
	  cout << " Will try to reconnect " << N << " more time";
	  if(N != 1) cout << "s";
	  cout << "... " << endl;
	  if(!isConnected())
	    {
	      cout<< " Waiting for "<< reconnectDelay_ 
		  << " secs before attempting to reconnect..." << endl;
	      usleep(1000000*reconnectDelay_); // convert to microsecs
	    }
	}
    }
}

// add socket to set of senders
// (i.e. set of sources if this is a collector, or set of collectors if this
// is a client)
void ClientRoot::addSender(string name)
{
  sources[currs] = SenderData();
  sources[currs].name = name;
}

// opposite action of connect method
void ClientRoot::disconnect(std::string host)
{
  // first, check if we this is one of the known hosts
  NameSocketMap::iterator it = serv_hostnames.find(host);
  if(it == serv_hostnames.end())
    {
      // host not known; maybe on list of pending connections?
      HostConAttemptMap::iterator it2 = collectorsPending.find(host);
      if(it2 != collectorsPending.end())
	// remove from set of pending connections
	collectorsPending.erase(it2);
      else
	{
	  cout << " *** Not connected to " << host 
	       << "! Request to disconnect ignored..." << endl;
	}
      // we are done
      return;
    }

  currs = it->second;
  const bool graceful_exit = true;
  // do not add node to collectorsPending
  const bool addNode2CollectorsPending = false;
  removeSocket(currs, graceful_exit, addNode2CollectorsPending); 
}

// disconnect from socket (if connected), release memory
void ClientRoot::disconnect(int sock) const
{
  if(isConnected(sock))
    // close socket if still connected
    close (sock);
}

// disconnect from all (connected or not) collectors
void ClientRoot::disconnect()
{
  collectorsPending.clear();
  for(SourceMap::iterator it = sources.begin(); 
      it != sources.end(); ++it)
    disconnect(it->first);
}

// true if connection with server was succesful
bool ClientRoot::isConnected(std::string host) const
{
  NameSocketMap::const_iterator it = serv_hostnames.find(host);
  if(it == serv_hostnames.end())
    return false;

  return isConnected(it->second);
}

// true if connection with server/collector was succesful
bool ClientRoot::isConnected(int s) const
{
  if (s >= 0)
    return true;
  return false;
}

// true if at least one connection was succesful
bool ClientRoot::isConnected() const
{
  return (!serv_hostnames.empty());
}

// true if all connections were succesful
bool ClientRoot::allConnected() const
{
  if(!collectorsPending.empty())
    return false; // at least one collector waiting to be connected

  return isConnected();
}

// set mask with active sockets
void ClientRoot::selectSock()
{
  // reset socket bit mask
  FD_ZERO (&rmask);                 
  currs = -1;

  if(multipleConnections())
    selectSockTMon();
  else
    selectSockSimple();
}

// set mask with  active sockets 
// to be used with single connection
void ClientRoot::selectSockSimple()
{
  if(isSource.empty())return;

  int sock = (isSource.begin())->first;
  FD_SET (sock, &rmask);
  
  struct timeval timeout = {0, timeout_*1000};
  int sfound = select (monMax + 1, &rmask, (fd_set *) 0, 
		       (fd_set *) 0, &timeout);

  if (sfound < 0) {
    if (errno == EINTR) 
      std::cerr << "interrupted system call\n" << std::endl;
    perror ("select");
    exit (1);
  }

}

// set mask with active sockets 
// to be used when having multiple connections
void  ClientRoot::selectSockTMon()
{
  int s = -1;
  //set all active sockets in mask
  std::map<int, bool>::iterator it;
  for (it = isSource.begin(); it != isSource.end(); ++it) 
    {
      FD_SET (it->first, &rmask);
    }
  FD_SET (ss, &rmask);
  
  // COLLECTORS
  if (timeout_ < 0)                                                 
    s = select (monMax + 1, &rmask, (fd_set *) 0, (fd_set *) 0, NULL);
  else
    {
      struct timeval timeout = {0, timeout_*1000};
      s = select (monMax+1, &rmask, (fd_set *) 0, (fd_set *) 0, &timeout);
    }  
  // notification for new source/client

  if (FD_ISSET (ss, &rmask)) 
    newConnection (ss);
}

// come here when server socket notifies of new connection
void ClientRoot::newConnection(int s)
{
  if(isSource.size() == MAX_CONN) 
    {
      cout << " Maximum # of connections reached (connection request ignored) "
	   << endl;
      return;
    }

  sockaddr_in *new_address = 0;
  socklen_t addrlen = sizeof (sockaddr_in);
  int s1 = accept (s, (sockaddr *) new_address, &addrlen);

  FD_SET (s1, &rmask);
  FD_CLR (ss, &rmask);

  SocketBoolMap::iterator it = isSource.find(s1);
  // check if connection already there
  if(it != isSource.end())
    {
      cout << " *** Attempt to start " << getHostname(s1)
	   << " connection ignored (connection already started)" << endl;
      disconnect(s1);
    }
  // check if client is not supposed to receive downstream connections
  else if(!isServer() && 
	  (serv_hostnames.find(getHostname(s1)) == serv_hostnames.end()) )
    {
      cout << " *** Refusing to serve monitoring requests from "
	   << getHostname(s1) << endl;
      disconnect(s1);
    }
  else
    addSocket(s1);
}

// add socket to isSource & set monMax
void ClientRoot::addSocket(int sock)
{
  isSource[sock] = true;       // assume source for now ("initial" value)
  // update monMax value, the new socket could have the higher fd number
  if (sock > monMax)           
    monMax = sock;

  cout << " Added socket connection at " << getHostname(sock) 
       << ", # of alive connections = " << isSource.size() << endl;  

}

// basic cleaning-up, applicable to both sources & clients
void ClientRoot::removeSocket_base(string name)
{
  all_names.erase(name);
  droppedNodes[getHostname(currs)] = name;
  isSource.erase(currs);
  disconnect(currs); // this call releases the TSocket memory
}

// remove source when disconnected;
// to be redefined by inheriting class
void ClientRoot::removeSocket(int s, bool graceful_exit, bool add2collPending)
{
  currs = s;
  senderIsDown(graceful_exit, add2collPending);
}

// come here when sender goes down; will try to reconnect (if applicable)
void 
ClientRoot::senderIsDown(bool graceful_exit, bool add2collPending)
{
  string name = sources[currs].name;
  cout << " Removing source " << name << "..." << endl;

  // directory cleanup
  setReceiverPtrs();
  ReceiverBase::cleanupSender();
  if(graceful_exit)
    ReceiverBase::senderIsDone();
  else
    ReceiverBase::senderIsDead();

  // hostname cleanup
  string hostname = getHostname(currs);
  serv_hostnames.erase(hostname);
  if(add2collPending)
    // assume time that server went down was same with last attempt to reconnect
    collectorsPending[hostname] = ConAttempt();

  sources.erase(currs);
  removeSocket_base(name);
  ++closedSources;
}

// b4 calling receiver: set addresses for socket, message & monitoring structure
void ClientRoot::setReceiverPtrs()
{
  ReceiverBase::recv_mess = mess;
  ReceiverBase::recv_socket = currs;
  ReceiverBase::sender_ = &(sources[currs]);
}

// check all pending server connections; 
// if problems try to reconnect (if applicable)
void ClientRoot::checkPendingConnections()
{
  // make sure there are connections that need to be established, and
  // we are supposed to attemp to reconnect
  if(collectorsPending.empty() || !shouldReconnect())return;

  for(HostConAttemptMap::iterator it = collectorsPending.begin(); 
      it != collectorsPending.end(); ++it)
    { // loop over pending server connections

      // check if we have tried too many times for this server
      bool tooManyTimes = (it->second.attempts == maxAttempts2Reconnect);
      if(tooManyTimes)
	{
	  cout << " *** Failed to recover collector at " << it->first << endl;
	  collectorsPending.erase(it);
	  throw (SenderData *) 0;
	}

      // check if it hasn't been long enough since last attempt
      TTimeStamp now;
      float dtime = now.GetSec() - it->second.when.GetSec();
      bool notTimeYet = (dtime < reconnectDelay_);
      if(notTimeYet)
	continue;
 
      // attempt to reconnect
      connect2Server(it->first);
    } // loop over pending server connections
}

// check if node has sent name & is identified as source; 
bool ClientRoot::newNode(string & buffer)
{
  if(newSource(buffer))
    return true;

  SocketBoolMap::iterator it = isSource.find(currs);
  // make sure socket belongs to isSource and that it corresponds to a source
  if(it != isSource.end() && it->second)
    // the message needs to be further processed in ReceiverBase
    // save the buffer
    ReceiverBase::buffer_ = &buffer;

  return false;
}


// check if node has sent name & is identified as source; 
// add to source list if appropriate
bool ClientRoot::newSource(const string & buffer)
{
  if(buffer.find(nameSourcePrefix) != string::npos)
    {
      // ------------------------------------------
      // Source has sent its name
      // ------------------------------------------
      TString name = buffer.c_str() + nameSourcePrefix.size();
      string sname = name.Data();
      string hostname = getHostname(currs);
      cout << " Added source identified as " << sname << " at "
	   << hostname << endl;
      
      bool need2cleanup = checkDroppedNodes(sname, hostname);
      insertName(sname);
      addSender(sname);

      isSource[currs] = true;
      if(serv_hostnames.find(getHostname(currs)) == serv_hostnames.end())
	ReceiverBase::introduceYourself(currs);
      if(need2cleanup)senderHasRecovered();
      return true;
    }

  return false;

}

// come here when (previously disconnected) sender has re-connected
void ClientRoot::senderHasRecovered()
{
  setReceiverPtrs();
  ReceiverBase::senderIsNotDead();
  ReceiverBase::senderIsNotDone();
  ReceiverBase::cleanupObsoleteSender();
}

// receive monitoring/send subscription requests; return success flag
bool ClientRoot::update()
{
  bool ret = true;
  try
    {
      if (mess) {delete mess; mess = new DQMMessage;}
      bool receivedUpdates = false;
      while(!receivedUpdates)
	{

	  // check if we need to attempt to connect to some server
	  checkPendingConnections();
	  
	  // first, try to retrieve monitorable and/or monitoring
	  receivedUpdates = receiveStuff();
	  if(mess)
	    {delete mess; mess = new DQMMessage;}

	  run_callbacks();

	  // here we attempt to send subscription requests to sources
	  doSourceLoop();

	  if(doNotBlockUpdate)
	    {
	      ret = receivedUpdates;
	      break;
	    }

	}

    }
  catch (SenderData * e)
    {
      if(!e)
	{
	  // will exit thread...
	  ret = false;
	}
    }

  
  delete mess; mess= new DQMMessage;
  
  return ret;
}

// attempt to receive monitoring, monitorable; 
// returns true if full monitoring set has been received
bool ClientRoot::receiveStuff(void)
{
  if (!selectCalled)
    {
      selectCalled = true;
      selectSock();
    }
  
  //now rmask is set with all the active sockets, just loop over all of them
  for (int fd = 0; fd <= monMax; ++fd)
    {
      if (fd == monMax) selectCalled = false;
      if (FD_ISSET (fd, &rmask))
	{
	  FD_CLR (fd, &rmask);
	  currs = fd; 
	  string buffer;
	  Int_t msize = SocketUtils::readMessage (mess, currs);
	  
	  if(conClosed(mess, msize, buffer))
	    {
	      // if this was a collector, we will add to list of pending connections
	      bool wasCollector = 
		serv_hostnames.find(getHostname(currs)) != serv_hostnames.end();
	      // if here: connection was dropped
	      removeSocket(currs, msize > 0, wasCollector);  
 	      return true; // true because this info is equivalent to new monitorable
	    }
  
	  if(mess->what() == kMESS_STRING)
	    // check for new node
	    if(newNode(buffer))
 	      return false; // false because there is no new monitorable yet
	  //  	      continue;
	  if(msize > 0)
	    {
	      bool received = unpackData ();
	      // all other messages should be handled by ReceiverBase (or SenderBase)
 	      return received; 
	    }
	  return false;
	}
    }
  return false;
}

// returns true if full monitoring set (or command) has been received
bool ClientRoot::unpackData()
{
  setReceiverPtrs();
  int result = ReceiverBase::receive();
  switch(result)
    {
    case kMESS_DONE_MONIT_CYCLE:
      sources[currs].cycles_count++;
      ++updates_;
      if(ReceiverBase::updates % 
	 ReceiverBase::printout_period(ReceiverBase::updates)== 0)
	summary();
      return true;
      //
    case kMESS_DONE_MONITORABLE:
      // return control if we have received full monitorable    
      return true;
      //
    default:
      // in all other cases return false
      return false;
    }
}

// true if <name_check> does not appear in all_names or droppedNodes
bool ClientRoot::unique_name(const string & name_check) const
{
  if(all_names.find(name_check) != all_names.end())
    return false;
  
  // this is slow... maybe reconsider logic if turns out to be a problem
  map<string, string>::const_iterator it;
  for(it = droppedNodes.begin(); it != droppedNodes.end(); ++it)
    if(name_check == (it->second))
      return false;

  // if here, it is a unique name
  return true;
}

// insert "name" to all_names; if name already exists, 
// insert modified "name" --> "name_n" instead (n: integer); 
// Will also check list of dropped nodes
void ClientRoot::insertName(string & name)
{
  string name_check = name;
  int n = 1;

  while(!unique_name(name_check))
    {
      ++n;
      std::ostringstream suffix; suffix << n;
      name_check = name + suffix.str();
    }
  
  if(strcmp(name.c_str(), name_check.c_str()))
    cout << " (Name exists; Renamed node to " << name_check << ") " << endl;

  // must insert & return the actual name used
  all_names[name_check] = currs;
  name = name_check; 
}

std::string ClientRoot::getHostname(int s) const
{
  sockaddr_in addr;
  socklen_t addrlen = sizeof (sockaddr_in);
  getpeername (s, (sockaddr *) &addr, &addrlen);	      
  hostent *hp = gethostbyaddr ((char *) &addr.sin_addr, 
			       sizeof (addr.sin_addr), addr.sin_family);
  char* hostname;

  if (hp == NULL)
    hostname = inet_ntoa (addr.sin_addr);
  else
    hostname = hp->h_name;
  
  return hostname;
}

// here we attempt to send subscription requests to sources
void ClientRoot::doSourceLoop(void)
{
  // loop over all sources
  for(SourceMap::iterator it = sources.begin(); it != sources.end(); ++it)
    {
      currs = it->first;
      setReceiverPtrs();
      ReceiverBase::doSubscription();
    } // loop over all sources
}

// return # of XXX instances in ROOT memory (specified by name)
int ClientRoot::getROOTcount(const TString & name) const
{
  gObjectTable->UpdateInstCount();
  
  TIter next(gROOT->GetListOfClasses());
  TClass *cl;
  int n = 0;//-999;
  while ((cl = (TClass*) next())) 
    {
      string cname = cl->GetName();
      if(cname.find(name.Data()) != string::npos)
	{
	  n = cl->GetInstanceCount();
	  break;
	}
    }
  
  return n;
}

#include "DQMServices/Core/interface/Tokenizer.h"

// printout of monitoring objects in memory
void ClientRoot::checkMemory()
{
  int i1 = getROOTcount("TH1F"); int i2 = getROOTcount("TH2F"); 
  int i3 = getROOTcount("TH3F"); int i4 = getROOTcount("TPro");
  int i5 = getROOTcount("TFol");
  cout << " ------------------------------------------------------------"<< endl;
  cout << " TH1F count = " << i1 << endl;
  cout << " TH2F count = " << i2 << endl;
  cout << " TH3F count = " << i3 << endl;
  cout << " TPro count = " << i4 << endl;
  cout << " TFol count = " << i5 << endl;
  cout << " ------------------------------------------------------------"<< endl;
  if(i1 == 0 && i2 == 0 && i3 == 0 && i4 == 0 && i5 == 0)
    {
      cout<<" Note: To check the memory footprint, you need to set these flags in"
	   << " ~/.rootrc" << endl;
      cout << " Root.MemStat:            1" << endl;
      cout << " Root.MemStat.size:       1" << endl;
      cout << " Root.MemStat.cnt:        1" << endl;
      cout << " Root.ObjectStat:         1" << endl;
      cout << " ------------------------------------------------------------" 
	   << endl;
    }
}

// will look for <hostname> in droppedNodes; return success;
// if found, modify sname to previous name if necessary
bool ClientRoot::checkDroppedNodes(string & sname, string hostname)
{
  bool ret = false;
  map<string, string>::iterator it = droppedNodes.find(hostname);
  if(it != droppedNodes.end())
    {
      if(it->second != sname)
	{
	  cout << " *** Warning! will use name from previously established"
	       << " connection : " << it->second << endl;
	  sname = it->second;
	}
      cout << " (Succesfully recovered previously disconnected node)" << endl;
      ret = true;
      droppedNodes.erase(it);
    }
  return ret;
}

// true if no upstream connection is made or pending
bool ClientRoot::noUpstreamConnections() const
{
  if(!isConnected() && collectorsPending.empty())
    return true;

  return false;
}

// run queue of callback methods
void ClientRoot::run_callbacks()
{
  LockMutex a(cb_queue.mutex);

  while(!cb_queue.cmds.empty())
    {
      // get next call back method
      seal::Callback action(cb_queue.cmds.front());
      // execute
      action(); 
      // remove action from queue
      cb_queue.cmds.pop();
    }


}

// add call back method; to be used for thread-unsafe operations
void ClientRoot::addCallback(seal::Callback & action)
{
  LockMutex a(cb_queue.mutex);
  cb_queue.cmds.push(action);
}

//return server socket; used to handle ctrl-c termination of collector process
int ClientRoot::getServerSocket ()
{
  return ClientRoot::ss;
}
