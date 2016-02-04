#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/WrapperNotifier.h"
#include "EventFilter/StorageManager/test/MockDiskWriterResources.h"

#include "xcept/tools.h"
#include "xdaq/ApplicationStub.h"
#include "xdaq/NamespaceURI.h"
#include "xdaq/WebApplication.h"
#include "xoap/domutils.h"
#include "xoap/MessageFactory.h"
#include "xoap/MessageReference.h"
#include "xoap/Method.h"
#include "xoap/SOAPBody.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPName.h"
#include "xoap/SOAPPart.h"


namespace stor {
    
    class fsmSoap : public xdaq::WebApplication
    {
        
    public:
        /**
         * Define factory method for the instantion of storage manager application.
         */
        XDAQ_INSTANTIATOR();
        
        /**
         * Constructor.
         */
        fsmSoap(xdaq::ApplicationStub *s) :
        xdaq::WebApplication(s)
        {
            std::cout << "Constructor" << std::endl;
            FragmentStore fs;
            SharedResourcesPtr sr;
            sr.reset(new SharedResources());
            sr->_initMsgCollection.reset(new InitMsgCollection());
            sr->_diskWriterResources.reset(new MockDiskWriterResources());
            sr->_commandQueue.reset(new CommandQueue(32));
            sr->_fragmentQueue.reset(new FragmentQueue(32));
            sr->_streamQueue.reset(new StreamQueue(32));
            
            EventDistributor ed(sr);

            WrapperNotifier wn(this);

            stateMachine = new StateMachine( &ed, &fs, &wn, sr );
            stateMachine->initiate();
            
	    xoap::bind(
                this,
                &stor::fsmSoap::processConfigureEvent,
                "Configure",
                XDAQ_NS_URI
            );
            
	    xoap::bind(
                this,
                &stor::fsmSoap::processEnableEvent,
                "Enable",
                XDAQ_NS_URI
            );
            
	    xoap::bind(
                this,
                &stor::fsmSoap::processStopEvent,
                "Stop",
                XDAQ_NS_URI
            );
            
	    xoap::bind(
                this,
                &stor::fsmSoap::processHaltEvent,
                "Halt",
                XDAQ_NS_URI
            );
	    
        };
        
        /**
         * Destructor.
         */
        ~fsmSoap()
        {
            // explicitly terminate the state machine before deleting it so
            // that we void the situation where a state object is trying to
            // access the state machine in its destructor, but the state
            // machine has already been destroyed
            stateMachine->terminate();

            delete stateMachine;
	    delete _app;

        };
        

        /**
         * Handle the "Configure" SOAP message
         */
        xoap::MessageReference processConfigureEvent(xoap::MessageReference msg) 
        {
            stateMachine->process_event( Configure() );
            
            return createResponseMessage("Configure");
        }
        

        /**
         * Handle the "Enable" SOAP message
         */
        xoap::MessageReference processEnableEvent(xoap::MessageReference msg) 
        {
            stateMachine->process_event( Enable() );
            
            return createResponseMessage("Enable");
        }
        

        /**
         * Handle the "Stop" SOAP message
         */
        xoap::MessageReference processStopEvent(xoap::MessageReference msg) 
        {
            stateMachine->process_event( Stop() );
            
            return createResponseMessage("Stop");
        }
        

        /**
         * Handle the "Halt" SOAP message
         */
        xoap::MessageReference processHaltEvent(xoap::MessageReference msg) 
        {
            stateMachine->process_event( Halt() );
            
            return createResponseMessage("Halt");
        }


        /**
         * Helper method to create the repsonse message
         */
        xoap::MessageReference createResponseMessage(std::string commandName)
        {
            xoap::MessageReference reply = xoap::createMessage();
            xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
            xoap::SOAPName responseName = envelope.createName( commandName + "Response", 
                "xdaq", XDAQ_NS_URI);
            (void) envelope.getBody().addBodyElement ( responseName );
            return reply;
        }


        /**
         * Unused method showing how to handle all state transitions in one method
         */
        xoap::MessageReference processFsmEvent(xoap::MessageReference msg) 
        {
            xoap::SOAPPart part = msg->getSOAPPart();
            xoap::SOAPEnvelope env = part.getEnvelope();
            xoap::SOAPBody body = env.getBody();
            DOMNode* node = body.getDOMNode();
            DOMNodeList* bodyList = node->getChildNodes();
            for (unsigned int i = 0; i < bodyList->getLength(); i++) 
            {
                DOMNode* command = bodyList->item(i);
                
                if (command->getNodeType() == DOMNode::ELEMENT_NODE)
                {
                    
                    std::string commandName = xoap::XMLCh2String (command->getLocalName());
                    
                    if (commandName != "Configure")
                        stateMachine->process_event( Configure() );
                    else if (commandName != "Enable")
                        stateMachine->process_event( Enable() );
                    else if (commandName != "Stop")
                        stateMachine->process_event( Stop() );
                    else if (commandName != "Halt")
                        stateMachine->process_event( Halt() );
                    else
                        XCEPT_RAISE(xoap::exception::Exception, "invalid command");

                    return createResponseMessage( commandName );
                }
            }
            
            XCEPT_RAISE(xoap::exception::Exception,"command not found");		
        }
        
    private:
        
      StateMachine *stateMachine;
      xdaq::Application* _app;
        
    };
    
}

/**
 * Provides the factory method for the instantiation of SM application.
 */
XDAQ_INSTANTIATOR_IMPL(stor::fsmSoap)

