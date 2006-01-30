#ifndef DQMSERVICES_DQMBASECLIENT_H
#define DQMSERVICES_DQMBASECLIENT_H

#include "DQMServices/Components/interface/StateMachine.h"

#include "xdata/include/xdata/UnsignedLong.h"
#include "xdata/include/xdata/String.h"
#include "xdata/include/xdata/Vector.h"

#include <string>

namespace dqm{
  class Updater;
}

class MonitorUserInterface;

class DQMBaseClient : public dqm::StateMachine
{

public:

  DQMBaseClient(xdaq::ApplicationStub *s, std::string name = "Client");
  virtual ~DQMBaseClient(){finalize();}
  void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  virtual void configure()=0;
  virtual void newRun()=0;
  virtual void endRun()=0;
  virtual void finalize(){};


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

  
  std::string name_;


  xdata::String server_;
  xdata::UnsignedLong port_;

  xdata::Vector<xdata::String> subs_;

};
#endif
