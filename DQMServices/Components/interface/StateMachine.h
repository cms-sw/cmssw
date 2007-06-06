#ifndef DQMSERVICES_COMPONENTS_STATEMACHINE
#define DQMSERVICES_COMPONENTS_STATEMACHINE

#include "EventFilter/Utilities/interface/EPStateMachine.h"
#include "xgi/WSM.h"

#include "xdaq/Application.h"

#include <string>

namespace log4cplus{
  class Logger;
}


namespace dqm
{
  class StateMachine : public xdaq::Application
  {
  public:
    StateMachine(xdaq::ApplicationStub *s);


    virtual ~StateMachine() {}

  public:

    ///transition methods to be implemented by the application
    virtual void configureAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;
    virtual void enableAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;
    virtual void stopAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception){}; //just a noop to keep the compiler happy for the moment
    virtual void suspendAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;
    virtual void resumeAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;
    virtual void haltAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;
    virtual void nullAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) = 0;


    xoap::MessageReference fireEvent(xoap::MessageReference msg)
      throw (xoap::exception::Exception);

  protected:

    void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
    {
      wsm_.displayPage(out);
    }
    
    void dispatch (xgi::Input * in, xgi::Output * out)  throw (xgi::exception::Exception)
    {
      cgicc::Cgicc cgi(in);
      cgi.getEnvironment();
      cgicc::const_form_iterator stateInputElement = cgi.getElement("StateInput");
      std::string stateInput = (*stateInputElement).getValue();
      wsm_.fireEvent(stateInput,in,out);
      fsm_.processFSMCommand(stateInput);
    }
    void bind(std::string);
    xdata::String *stateName(){return &fsm_.stateName_;}

  private:

    void statePage( xgi::Output * out ) 
      throw (xgi::exception::Exception);

    void failurePage(xgi::Output * out, xgi::exception::Exception & e)  
      throw (xgi::exception::Exception);

    /**
       Web Events that trigger state changes 
    */
    void webConfigure(xgi::Input * in ) throw (xgi::exception::Exception)
    {
      
    }
    
    void webEnable(xgi::Input * in ) throw (xgi::exception::Exception)
    {
      
      
    }
    void webSuspend(xgi::Input * in ) throw (xgi::exception::Exception)
    {
      
      
    }
    void webResume(xgi::Input * in ) throw (xgi::exception::Exception)
    {
      
      
    }
    void webHalt(xgi::Input * in ) throw (xgi::exception::Exception)
    {
      
      
    }
	


    xgi::WSM wsm_;
    evf::EPStateMachine fsm_;
    std::string page_;
  };
}
#endif
