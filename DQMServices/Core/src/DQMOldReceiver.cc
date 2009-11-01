#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMNet.h"
#include <iostream>

// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
// using <client_name>; if flag=true, will accept downstream connections
DQMOldReceiver::DQMOldReceiver(const std::string &hostname, int port,
			       const std::string &clientName,
			       int unusedReconnectDelaySecs,
			       bool unusedActAsServer)
  : store_ (DQMStore::instance()),
    net_ (new DQMBasicNet (clientName)),
    numUpdates_ (0)
{
  net_->requestFullUpdates(true);
}

/* Use the default constructor for running in standalone mode (ie. without
   sources or collectors); if flag=true, client will accept downstream connections
*/
DQMOldReceiver::DQMOldReceiver(void)
  : store_ (DQMStore::instance()),
    net_ (0),
    numUpdates_ (0)
{}

DQMOldReceiver::~DQMOldReceiver(void)
{
  delete net_;
}

// this is the "main" loop where we receive monitoring/send subscription requests;
// if client acts as server, method runQTests is also sending monitoring & 
// test results to clients downstream;
// returns success flag
bool
DQMOldReceiver::update(void)
{
  std::cout
    << " In DQMOldReceiver::update:\n"
    << " This method will be deprecated soon, please replace mui->update() by:\n"
    << "     bool ret = mui->doMonitoring();\n"
    << "     bei->runQTests();\n";

  // retrieval of monitoring, sending of subscription requests/cancellations,
  // calculation of "collate"-type Monitoring Elements;
  bool ret = doMonitoring();

  // Run quality tests (and determine updated contents);
  // Method is overloaded if client acts as server to other clients downstream
  store_->runQTests();

  return ret;
}

// retrieval of monitoring, sending of subscription requests/cancellations,
// returns success flag
bool
DQMOldReceiver::doMonitoring(void)
{
  store_->reset();
  numUpdates_ += (net_ ? net_->receive(store_) : 0);
  // FIXME: Invoke callbacks?
  return true;
}
