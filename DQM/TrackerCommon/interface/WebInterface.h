#ifndef _DQM_TrackerCommon_WebInterface_h_
#define _DQM_TrackerCommon_WebInterface_h_

#include "xgi/Method.h"
#include "xdata/UnsignedLong.h"
#include "cgicc/HTMLClasses.h"

#include "xdaq/Application.h"
#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "DQM/TrackerCommon/interface/WebPage.h"
#include "DQM/TrackerCommon/interface/MessageDispatcher.h"
#include "DQM/TrackerCommon/interface/ME_MAP.h"

#include <string>
#include <map>

class DQMStore;

class WebInterface
{

 private:

  std::string exeURL;
  std::string appURL;
  std::multimap<std::string, std::string> conf_map;

  MessageDispatcher msg_dispatcher;

 protected:

  DQMStore* dqmStore_;
  WebPage * page_p;

 public:
  
  WebInterface(std::string _exeURL, std::string _appURL);
  virtual ~WebInterface() 
    {
    }

  std::string getContextURL()     { return exeURL; }
  std::string getApplicationURL() { return appURL; }

  void handleRequest        (xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void handleStandardRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  virtual void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
    {
    };

  /// Answers requests by sending the webpage of the application
  virtual void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  /// Answers connection configuration requests
  void Configure(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  
  /// Answer navigator requests 
  void Open(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
//  void Subscribe(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
//  void Unsubscribe(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);

  /// Outputs the subdirectories and files of "directory". Called by any of the above three.
  void printNavigatorXML(std::string directory, xgi::Output * out);
  
  /// Answers ContentViewer requests
  void ContentsOpen(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void printContentViewerXML(std::string current, xgi::Output * out);

  /// Answers Messages requests
  void printMessagesXML(xgi::Output *out);

  /// Answers viewer requests
  void DrawGif(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void printMap(ME_MAP view_map, std::string id);

  /// Adds widgets to the page
  void add(std::string, WebElement *);

  /// Adds messages to the message dispatcher
  void sendMessage(std::string the_title, std::string the_text, MessageType the_type);

  std::string get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key);

};

#endif
