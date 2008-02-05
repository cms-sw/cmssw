#ifndef ClientServerRoot_H
#define ClientServerRoot_H

#include "DQMServices/Core/src/ClientRoot.h"
#include "DQMServices/Core/src/SenderBase.h"

#include <string>
#include <vector>
#include <map>


class ClientServerRoot : public ClientRoot, public SenderBase
{
 public:
  /// ctor #1: configure client w/ collector hostname & port it should connect to
  ClientServerRoot(std::string host,unsigned port, std::string client_name,
		   bool keepStaleSources, bool sendLocalMEsOnly);
  /// ctor #2: client configuration for connection w/ multiple collectors
  ClientServerRoot(std::vector<std::string> hosts, unsigned port, 
		   std::string client_name, bool keepStaleSources,
		   bool sendLocalMEsOnly);
  virtual ~ClientServerRoot();

  /// get # of connected sources, clients
  unsigned getNumSourcesOn(void) const {return sources.size();}
  unsigned getNumClientsOn(void) const {return clients.size();}
  /// get # of disconnected sources, clients
  unsigned getNumSourcesOff(void) const {return closedSources;}
  unsigned getNumClientsOff(void) const {return closedClients;}
  /// get # of names for connected sources, clients
  std::vector<std::string> getSourceNames(void) const;
  std::vector<std::string> getClientNames(void) const;
  /// get host name for node (source or client)
  std::string getHostName(const std::string & node_name) const;
  /// get # of monitoring packages received
  unsigned getNumReceived(const std::string & source_name) const;
  /// get # of monitoring packages sent
  unsigned getNumSent(const std::string & client_name) const;
  /// get total #'s of monitoring packages received, sent
  unsigned getNumReceived(void) const{return ReceiverBase::updates;}
  unsigned getNumSent(void) const{return SenderBase::updates;}


 protected:
  /// attempt to send monitoring, quality test results
  void sendStuff();
  /// attempt to send monitoring, quality test results
  void doClientLoop();
  /// # of disconnected clients
  unsigned closedClients;
  /// returns true if full monitoring set (or command) has been received
  bool unpackData();
  /// b4 calling sender: set addresses for socket, 
  /// message & monitoring structure
  void setSenderPtrs();
  /// come here to remove a client
  void removeClient(int s);
  /// show statistics
  void showStats();
  /// check for new source/client subscribing; return success flag
  bool newNode(std::string & buffer);
  /// check if node has sent name & is identified as client; 
  /// add to client list if appropriate
  bool newClient(const std::string & buffer);
  /// add socket to set of receivers/clients
  void addReceiver(std::string name);
  /// check whether all receivers are done sending
  bool allReceiversDone();
  /// check whether all senders are done sending
  bool allSendersDone();
 
 private:
  /// to be called by ctors
  void init();

  /// key: socket fd, value: client
  typedef std::map<int, ReceiverData> ClientMap;
  /// map containing data for all downstream clients
  /// add items with addClient; remove items with removeClient
  ClientMap clients;

  /// remove node when disconnected;
  void removeSocket(int s, bool graceful_exit, bool add2collPending);
  /// printout stats and check for memory leaks
  void summary(bool checkMemoryLeaks = false);
  /// come here to determine if client has gone down
  void checkClient();
  /// maximum # of consecutive failures in sending allowed
  int s_fail_consec_max;

};

#endif
