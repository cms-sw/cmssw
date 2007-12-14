#ifndef DQMSERVICES_DQMBASECLIENT_H
#define DQMSERVICES_DQMBASECLIENT_H

#include "DQMServices/XdaqCollector/interface/StateMachine.h"

#include "xdata/UnsignedLong.h"
#include "xdata/String.h"
#include "xdata/Vector.h"
#include "xdata/Boolean.h"


#include <string>

namespace dqm{
  class Updater;
}

class MonitorUserInterface;

class DQMBaseClient : public dqm::StateMachine
{

public:

  DQMBaseClient(xdaq::ApplicationStub *s, 
		std::string name = "DQMBaseClient", 
		std::string server = "localhost", 
		int port = 9090, 
		int reconnect_delay_secs = 5,
		bool actAsServer = false);
  void fireConfiguration(std::string name, std::string server, int port);
  virtual ~DQMBaseClient(){finalize();}
  void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  virtual void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  virtual void configure()=0;
  virtual void newRun()=0;
  virtual void endRun()=0;
  virtual void finalize(){};

  std::string getApplicationURL() 
    {
      return applicationURL;
    }
  std::string getContextURL()
    {
      return contextURL;
    }

protected:

  MonitorUserInterface *mui_; 
  dqm::Updater *upd_;

private:

  void configureAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);
  
  void enableAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);
    
  void suspendAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void resumeAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void haltAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void nullAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  std::string contextURL;
  std::string applicationURL;

  std::string name_;

  xdata::String server_;

  xdata::UnsignedLong port_;
  xdata::UnsignedLong reconnect_delay_secs_;

  xdata::Boolean actAsServer_;

  xdata::Vector<xdata::String> subs_;

};
#endif
