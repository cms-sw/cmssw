#ifndef RPCWebClient_H
#define RPCWebClient_H

/** \class RPCWebClient
 * *
 *  Client Class that performs the handling of monitoring elements for
 *  the RPC Data Quality Monitor (mainly prints the control web page and runs quality tests
 *  on real-time demand).
 * 
 *  $Date: 2006/01/30 17:38:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/WebComponents/interface/DQMWebClient.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQM/RPCMonitorClient/interface/RPCQualityTester.h"

class Button;

class RPCWebClient : public DQMWebClient
{
 public:
  
  XDAQ_INSTANTIATOR();
  
  explicit RPCWebClient(xdaq::ApplicationStub * s);

  /// The method that prints out the webpage
  void Default(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  
  ///A method that responds to WebElements submitting non-default requests (like Buttons)
  void Request(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
 
  /// Set up Quality Tests
  void ConfigQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  
  /// Run Quality Tests
  void RunQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  
  /// Check Status of Quality Tests
  void CheckQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  
  /// Check Status of Quality Tests
  void CheckQTestsRequestSingle(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);

 private:

  WebPage * page;
  bool printout;
  bool testsWereSet;
  
  
  RPCQualityTester * qualityTests;

  Button * butRunQT;
  Button * butCheckQT;
  Button * butCheckQTSingle;
  
  int yCoordinateMessage;
};

#endif
