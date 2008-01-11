#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/interface/MonitorData.h"
#include "DQMServices/Core/interface/RootMonitorThread.h"
#include "DQMServices/Core/interface/SocketUtils.h"
#include "DQMServices/Core/interface/DQMMessage.h"
#include "DQMServices/Core/interface/DQMShipMonitoring.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <errno.h> 

#include <iostream>
#include <string>

using namespace dqm::me_util;
using namespace dqm::monitor_data;

using std::cout; using std::endl; using std::cerr;
using std::string;

const bool RootMonitorThread::resetMEs = true; 
const bool RootMonitorThread::callResetDiff = true;

// del: time delay in between shipments (in microsecs)
RootMonitorThread::RootMonitorThread(string host, unsigned port, unsigned del,
				     string name, 
				     edm::ServiceToken iToken,
				     int recon_del)
  : SenderBase(name, del, true), lock(0), recv_name("Collector"), 
    terminate_(false), token_(iToken), 
    maxAttempts2Reconnect(MAX_RECON)
{
  NodeBase::host_ = host; NodeBase::port_ = port;
  SenderBase::bei = edm::Service<DaqMonitorBEInterface>().operator->();


  edm::ServiceRegistry::Operate operate(token_);

  cout << " Service DQMShipMonitoring has been ";
  // if DQMShipMonitoring service is enabled, will ship monitoring in same thread
  if(edm::Service<DQMShipMonitoring>().isAvailable() )
    {
      cout << "enabled" << endl;
      shipMonitoringIn2ndThread = false;
    }
  else
    {
      cout << "disabled" << endl;
      shipMonitoringIn2ndThread = true;
    }

  mess = new DQMMessage;
  s_fail_consec_max = 50; sock = -1; maxSock = -1; 
  pause(); // Thread method
  connect(); // attempt to connect
  setReconnectDelay(recon_del);
  initialize(); // Thread method
}

// true if connection was succesful
bool RootMonitorThread::isConnected(void) const
{
  if (sock >= 0)
    return true;
  return false;
}


// come here to connect once host & port have been determined
void RootMonitorThread::connect(void)
{
  if (isConnected ())
    {
      cerr << " *** Connection already established! Connection request ignored..." << endl;
      return;
    }
  sock = NodeBase::connect(nameSourcePrefix);
  // set receiver folder
  if (isConnected()) 
    {
      // set receiver folder
      receiver.Dir = SenderBase::makeDirStructure(recv_name);
      receiver.name = recv_name;
      if (sock > maxSock)
	maxSock = sock;
    }
}

// close connection
void RootMonitorThread::closeConnection(void)
{
  if(isConnected())
    {
 	SocketUtils::sendString (Quitting.c_str(), sock);
	close(sock);
    }
  // release memory
  if (sock >= 0) {sock = -1;}
}

RootMonitorThread::~RootMonitorThread(void)
{
  if(!terminate_)
    terminate();

  finalize();
  cout << " Thread "<< self_tid() << " has been terminated!" << endl;
}

// return connection status
bool RootMonitorThread::checkConnection(void)
{
  // check if we are connected
  if(!isConnected())
    {
      // if not, check whether we should try to connect
      if(!shouldReconnect())
	return false; // not instructed to reconnect; return failure
 
      wait_and_connect();
    }

  // done trying
  return isConnected();
}

// infinite monitoring loop
void RootMonitorThread::run(void)
{ 
  if(!checkConnection())return;

  // make the services active
  edm::ServiceRegistry::Operate operate(token_);

  edm::Service< DaqMonitorBEInterface > bei;
  *bei; 
 
  while (1)
    {
      SimpleLockMutex gl(genLock);
      if(terminate_) 
	{
	  closeConnection();
	  break;
	}

      try
	{
	  setSenderPtrs();

	  receiveStuff(100); // wait time: 100 msecs
	  // Note: this should be a configuration parameter...
	  
	  // make sure receiver is not in the middle of sending something
	  if(isReceiverDone())
	    {
	      sendStuff();
	    }

	}
      catch(ReceiverData * r)
	{
 	  if(r)
	    cout << " Succesfully recovered collector... " << endl;
	  else
	    {
	      cout << " Failed to recover collector..." << endl;
	      terminate_ = true;
	    }
	}
    } // infinite loop

}

// lock
void RootMonitorThread::pause()
{
  if(lock==0)
    lock = new boost::mutex::scoped_lock(genLock);
}

// unlock
void RootMonitorThread::release()
{
  if(lock)
    delete lock;
  lock = 0;
}

// it seems that this is called by destructor (ie. does not need explicit call)
void RootMonitorThread::terminate()
{
  terminate_ = true;
  // wait for lock (indicates that secondary thread has closed connection)
  SimpleLockMutex gl(genLock);
}

void RootMonitorThread::abort()
{
  cout << " Thread "<< self_tid() << " aborted!" << endl;
}

// b4 calling sender: set addresses for socket and monitoring structure
void RootMonitorThread::setSenderPtrs(void)
{
  SenderBase::send_socket = sock;
  SenderBase::send_mess = mess;
  SenderBase::receiver_ = &receiver;
}

// false if collector is not responsive
bool RootMonitorThread::isCollectorAlive(void)
{
  if(receiver.n_failed)
    {
      if(receiver.n_failed_consec % s_fail_consec_max == 0)
	{
	  cout << " *** Failed to communicate with node " 
	       << receiver.name << " " 
	       << receiver.n_failed_consec << " consecutive times"
	       << endl;
	  return false;
	}
    }
  // if here, collector should be alive...
  return isConnected();
}

// attempt connection if previous call failed
void RootMonitorThread::connect(string host, unsigned port)
{
  NodeBase::host_ = host; NodeBase::port_ = port;
  connect();
}

// come here when collector has died; try to resurrect connection
void RootMonitorThread::recoverCollector(void)
{
  // try to reconnect
  wait_and_connect();

  // if connected:
  // set newNode to true, in order to resend monitorable
  if(isConnected())receiver.newNode = true;
}

// come here when collector has died; recover connection, or return
void RootMonitorThread::collectorIsDead(void)
{
  bei->unlock();

  // first, do some cleanup
  setSenderPtrs();
  SenderBase::cleanupReceiver();

  // then try to reconnect
  if(shouldReconnect())
    recoverCollector();

  if(isConnected())
    throw (ReceiverData *) SenderBase::receiver_;
  else
    throw (ReceiverData *) 0;
}

// send monitoring only (to be called by DQMShipMonitoring)
void RootMonitorThread::sendMonitoringOnly()
{
  setSenderPtrs();
  if(!isConnected())return;

  SenderBase::startSending();
  SenderBase::shipMonitoring();
  SenderBase::doneSendingMonitoring(resetMEs, callResetDiff);
}

// sending monitorable & monitoring
void RootMonitorThread::sendStuff(void)
{
  int N = 0; // # of monitoring objects sent

  SenderBase::startSending();

  if(shipMonitoringIn2ndThread)
    {
      N = SenderBase::send();
      SenderBase::doneSending(resetMEs, callResetDiff);
    }
  else
    {
      SenderBase::shipMonitorable();
      SenderBase::doneSendingMonitorable(callResetDiff);
    }

  if(N < 0 && !isCollectorAlive())
    collectorIsDead();
}


// receive (un)subscription requests (wait up to <wait_msecs> msecs)
void RootMonitorThread::receiveStuff(int wait_msecs)
{
  // reset all bit in the map (just one socket is in it)
  FD_ZERO (&rmask);                                             
  //set the receiving socket  in order to check its status when calling select
  FD_SET (sock, &rmask);                                        
  int msize = -1; string buffer;
  struct timeval timeout = {0, wait_msecs*1000};
  int sfound = select (maxSock + 1, &rmask, (fd_set *) 0, (fd_set *) 0, &timeout);
  
  if (sfound < 0) {
    if (errno == EINTR) 
      std::cerr << "interrupted system call\n" << std::endl;
    perror ("select");
    exit (1);
  }

  if (FD_ISSET (sock, &rmask))
      msize = SocketUtils::readMessage (mess, sock);
  else
    {
      if(mess)
	delete mess; 
      mess = new DQMMessage;
    }

  if(conClosed(mess, msize, buffer))
    collectorIsDead();
  
  SenderBase::buffer_ = &buffer;
  setSenderPtrs();
  SenderBase::getSubscription();
}


// wait <reconnectDelay_> secs, then attempt to connect, 
// up to maxAttempts2Reconnect times
void RootMonitorThread::wait_and_connect()
{
  // first close connection before re-connection attempt
  if(sock >= 0) {
    close (sock);
    sock = -1;
  }
  
  unsigned attempts = 0;
  while(!isConnected() && (attempts < maxAttempts2Reconnect))
    {
      cout<< " Waiting for "<< reconnectDelay_ 
	  << " secs before attempting to reconnect..." << endl;
      usleep(1000000*reconnectDelay_); // convert to microsecs
      // stop trying if terminate() has been called
      if(terminate_)break;

      this->connect();
      ++attempts;
      int n = maxAttempts2Reconnect - attempts;
      if(n)
	{
	  cout << " Will try to reconnect " << n << " more time";
	  if(n != 1) cout << "s";
	  cout << "... " << endl;
	}

    }

}
