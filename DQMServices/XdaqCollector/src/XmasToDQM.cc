// $Id: Application.cc,v 1.2 2008-01-18 16:21:56 oh Exp $

/*************************************************************************
 * XDAQ Components for Distributed Data Acquisition                      *
 * Copyright (C) 2000-2004, CERN.			                 *
 * All rights reserved.                                                  *
 * Authors: J. Gutleber and L. Orsini					 *
 *                                                                       *
 * For the licensing terms see LICENSE.		                         *
 * For the list of contributors see CREDITS.   			         *
 *************************************************************************/
#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTMLClasses.h"
#include "cgicc/HTTPResponseHeader.h" 


#include "xdaq/ApplicationGroup.h" 
#include "xdaq/ApplicationRegistry.h" 

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPPart.h"
#include "xoap/SOAPBody.h"
#include "xoap/SOAPBodyElement.h"
#include "xoap/Method.h"
#include "xoap/domutils.h"
#include "xoap/DOMParser.h"
#include "xoap/SOAPHeader.h"
 
#include "xgi/Table.h" 
#include "xcept/tools.h"

#include "DQMServices/XdaqCollector/interface/XmasToDQM.h"
//#include "xplore/DiscoveryEvent.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "xdata/InfoSpaceFactory.h"
//#include "xplore/Interface.h"
//#include "xplore/exception/Exception.h"
//#include "ws/addressing/WSAddressing.h"
//#include "xdata/exdr/Serializer.h"
#include "toolbox/Runtime.h"


#include "xdata/exdr/FixedSizeInputStreamBuffer.h"
#include "xdata/Table.h"
#include "xdata/exdr/Serializer.h"
#include "xdata/exdr/AutoSizeOutputStreamBuffer.h"
#include "xdaq/ApplicationDescriptorImpl.h"

#include "toolbox/TimeVal.h"
#include "toolbox/stl.h"
#include "toolbox/regex.h"

//#include "ws/eventing/Identifier.h"
//#include "ws/eventing/WSEventing.h"
//#include "ws/eventing/SubscribeRequest.h"
//#include "DQMServices/XmasToDQM/interface/SubscribeRequest.h"
//#include "ws/eventing/RenewRequest.h"
//#include "xmas/xmas.h"
#include "xoap/Event.h"

#include<boost/tokenizer.hpp>
	
	

XDAQ_INSTANTIATOR_IMPL(xmas2dqm::wse::XmasToDQM)

xmas2dqm::wse::XmasToDQM::XmasToDQM(xdaq::ApplicationStub* s)  throw (xdaq::exception::Exception) 
	: xdaq::Application(s)
{	
	getApplicationDescriptor()->setAttribute("icon", "/xmas2dqm/wse/images/Las.png");
	
	LOG4CPLUS_INFO(this->getApplicationLogger(),"inside constructor of xmas2dqm::wse::Application");

	
	// Activates work loop for las asynchronous operations (SOAP messages)
	//dispatcher_.addActionListener(this);
	//(void) toolbox::task::getWorkLoopFactory()->getWorkLoop("urn:xdaq-workloop:las", "waiting")->activate();
	
	
	// bind SOAP interface
	xoap::bind(this, &xmas2dqm::wse::XmasToDQM::fireEvent, "Enable", XDAQ_NS_URI );
	xoap::bind(this, &xmas2dqm::wse::XmasToDQM::fireEvent, "Halt", XDAQ_NS_URI );
	xoap::bind(this, &xmas2dqm::wse::XmasToDQM::reset, "Reset", XDAQ_NS_URI );
	
	
	// Define FSM
	//		
	fsm_.addState('H', "Halted", this, &xmas2dqm::wse::XmasToDQM::stateChanged);
	fsm_.addState('E', "Enabled", this, &xmas2dqm::wse::XmasToDQM::stateChanged);

	fsm_.addStateTransition('H', 'E', "Enable", this,&xmas2dqm::wse::XmasToDQM::EnableAction);
	fsm_.addStateTransition('H', 'H', "Halt", this, &xmas2dqm::wse::XmasToDQM::HaltAction);
	fsm_.addStateTransition('E', 'H', "Halt", this, &xmas2dqm::wse::XmasToDQM::HaltAction);

	// Failure state setting
	fsm_.setFailedStateTransitionAction( this, &xmas2dqm::wse::XmasToDQM::failedTransition );
	fsm_.setFailedStateTransitionChanged(this, &xmas2dqm::wse::XmasToDQM::stateChanged );

	fsm_.setInitialState('H');
	fsm_.setStateName('F', "Failed"); // give a name to the 'F' state

	fsm_.reset();

	// Export a "State" variable that reflects the state of the state machine
	state_ = fsm_.getStateName (fsm_.getCurrentState());
	getApplicationInfoSpace()->fireItemAvailable("stateName",&state_);
	getApplicationInfoSpace()->fireItemAvailable("LASurl",&LASurl_);
	getApplicationInfoSpace()->fireItemAvailable("Period",&Period_);
	getApplicationInfoSpace()->fireItemAvailable("LASQueueSize",&LASQueueSize_);
	
	// Add infospace listeners for exporting data values
	getApplicationInfoSpace()->addItemChangedListener ("stateName", this);
	getApplicationInfoSpace()->addItemChangedListener ("LASurl", this);
	getApplicationInfoSpace()->addItemChangedListener ("Period", this);
	getApplicationInfoSpace()->addItemChangedListener ("LASQueueSize", this);
	
	LASurl_ = "http://srv-c2d04-18.cms:9943/urn:xdaq-application:lid=100/retrieveCollection";
	Period_ = "10";
	LASQueueSize_ = "100000";
	
	//http://srv-c2d04-18.cms:9943/urn:xdaq-application:lid=100/retrieveCollection
	//http://fu16.cmsdaqpreseries:9943/urn:xdaq-application:lid=100/retrieveCollection
	
	
	//curl_global_init(CURL_GLOBAL_DEFAULT);
    	curl_global_init(CURL_GLOBAL_ALL);
	
	LASReadout_ = toolbox::task::bind (this, &xmas2dqm::wse::XmasToDQM::LASReadoutWorkLoop, "LASReadoutWorkLoop");
	
	LASReadoutWorkLoop_ = toolbox::task::getWorkLoopFactory()->getWorkLoop("LASReadoutWaitingWorkLoop", "waiting");
	
	
	if (LASReadoutWorkLoop_->isActive() == false) 
	{
		LASReadoutWorkLoop_->activate();
		
	}
	
	//LOG4CPLUS_INFO(this->getApplicationLogger(),"insdie constructor of xmas2dqm::wse::XmasToDQM Period = " << Period_.toString());
	
	LASReadoutTimer_ = toolbox::task::getTimerFactory()->createTimer("PeriodicLASReadout");

        // toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
//         
// 	startLASReadout_ = toolbox::TimeVal::gettimeofday();
//         
// 	LASReadoutTimer_->scheduleAtFixedRate( startLASReadout_, this, interval,  0, std::string("LASReadout") );	

	LOG4CPLUS_INFO(this->getApplicationLogger(),"finish of Constructor of xmas2dqm::wse::XmasToDQM");
	
}


bool xmas2dqm::wse::XmasToDQM::LASReadoutWorkLoop (toolbox::task::WorkLoop* wl)
{
	//keep log how many times the workloop was called...
	static int times = 0;
	times++;
	
	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...ready to ask LAS for EXDR data : time " << times <<"th");
	LOG4CPLUS_INFO (getApplicationLogger(), "Period = " + Period_.toString() + " LASurl = " + LASurl_.toString());
	
	
	xdata::Table * ptr_table = new xdata::Table();
	
	int ret = getEXDR_LAS(ptr_table);
	
	LOG4CPLUS_INFO (getApplicationLogger(), "return value from getEXDR_LAS = " << ret );
	
	if (ret == -1 || ret == -2) //exception during xdata::exdr deserialization occured or during CURL (could log different messages)
	{	
		//LOG4CPLUS_INFO (getApplicationLogger(), "LASWorkLoop ready to call xdata::Table::Reference destructor..." );
		
		//ref_table->~Table();
		
		
		LOG4CPLUS_INFO (getApplicationLogger(), "LASWorkLoop freeing xdata::Table * space" );
		
		delete ptr_table;
		
		LOG4CPLUS_WARN (getApplicationLogger(), "getEXDRLAS didn't complete properly, returning from LASREeadoutWorkLoop" );
		return false;
	}

	
	
	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...lock the mutex ");
	
	
	//BSem_.take();
	//acquire the mutex - protect access to the queue
	pthread_mutex_lock(&xmas2dqm::wse::ToDqm::instance()->mymutex_);
   
   	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...check (...and possible wait) if data queue is full");
	
	//check if the queue is full and wait (a signal that informs that an element has been poped)
	// until there is 'space' in the queue    
    	while (xmas2dqm::wse::ToDqm::instance()->MemoryTable_.size() >= atoi(LASQueueSize_.toString().c_str())/*Qsize_max*/)
	{
        	pthread_cond_wait(&xmas2dqm::wse::ToDqm::instance()->less_, &xmas2dqm::wse::ToDqm::instance()->mymutex_);
	}
	
	LOG4CPLUS_DEBUG (getApplicationLogger(), "data queue has available store...proceeding...");
	
		
	
	std::map<std::string, std::string, std::less<std::string> >::iterator i;
	
	//size_t row = ref_table->getRowCount();
	// size_t row = ptr_table->getRowCount();
// 
// 	for ( size_t r = 0; r <  ptr_table->numberOfRows_ /*ref_table->numberOfRows_*/; r++ )
// 	{
// 	    	LOG4CPLUS_INFO(this->getApplicationLogger(),"********* Printing table inside XmasToDQM ***************");
// 		LOG4CPLUS_INFO(this->getApplicationLogger(),/*ref_table*/ ptr_table->columnData_["bxHistogram"]->elementAt(r)->toString());
// 	    
// 	    	boost::tokenizer<> tokens(/*ref_table*/ ptr_table->columnData_["bxHistogram"]->elementAt(r)->toString());
//    	    	for(boost::tokenizer<>::iterator itok=tokens.begin(); itok!=tokens.end();++itok)
// 	    	{
//        			//LOG4CPLUS_INFO(this->getApplicationLogger, (std::string)(*itok) );
//    		}
// 	    
// 	    	row++;
// 	}
	
	
	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...ready to call XmastoDQM::ToDQM::digest " );
	
	//insert LAS data table to ToDQM objec - queue
	xmas2dqm::wse::ToDqm::instance()->digest("flashListName", "originator", "tagName", ptr_table /*ref_table*/);
	
	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...signaling new element added to the data queue ");
	
	//signal that a new element has been inserted
	pthread_cond_signal(&xmas2dqm::wse::ToDqm::instance()->more_);

	LOG4CPLUS_INFO (getApplicationLogger(), "inside WorkLoop...release mutex, allow access to the data queue");

	//allow access to the queue
    	pthread_mutex_unlock(&xmas2dqm::wse::ToDqm::instance()->mymutex_);	
	
	return false;
}



int xmas2dqm::wse::XmasToDQM::getEXDR_LAS(/*xdata::Table::Reference & rtable*/ xdata::Table * rtable)
{
	LOG4CPLUS_INFO(this->getApplicationLogger(),"inside getEXDR_LAS.........");	
	
    	CURL *curl_handle;
	CURLcode code;
	 
  	char *data="fmt=exdr&flash=urn:xdaq-flashlist:frlHisto";
	
    	struct MemoryStruct chunk;
  
   	chunk.memory=NULL; /* we expect realloc(NULL, size) to work */
   	chunk.size = 0;    /* no data at this point */
	  
    	/* init the curl session */
    	curl_handle = curl_easy_init();
  
  	if (curl_handle == NULL)
    	{
      		LOG4CPLUS_INFO(getApplicationLogger(), "Failed to create CURL connection");
  
      		return -2;
    	}  	
      
    	code = curl_easy_setopt(curl_handle, CURLOPT_POSTFIELDS, data);
    	if (code != CURLE_OK)
    	{
      		LOG4CPLUS_INFO (getApplicationLogger(),"Failed to set post fields");
  
     		return -2;
   	}
	
	/* specify URL to get */
    	curl_easy_setopt(curl_handle, CURLOPT_URL, LASurl_.toString().c_str()/*"http://fu16.cmsdaqpreseries:9943/urn:xdaq-application:lid=100/retrieveCollection"*/);
  
    	/* send all data to this function  */
    	curl_easy_setopt(curl_handle, /*CURLOPT_HEADERFUNCTION*/ CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
	
	//curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, HeaderMemoryCallback);
  
    	/* we pass our 'chunk' struct to the callback function */
   	curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
 
   	/* some servers don't like requests that are made without a user-agent
      	field, so we provide one */
   	curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
 
   	/* get it! */
   	curl_easy_perform(curl_handle);
 
   	/* cleanup curl stuff */
   	curl_easy_cleanup(curl_handle);
 
  	 /*
    	* Now, our chunk.memory points to a memory block that is chunk.size
    	* bytes big and contains the remote file.
    	*
    	* Be aware of the fact that at this point we might have an
    	* allocated data block, and nothing has yet deallocated that data. So when
    	* you're done with it, you should free() it as a nice application.
    	*/
 
 	LOG4CPLUS_INFO (getApplicationLogger(),"chunk.memory length = " << chunk.size);
 	
	xdata::exdr::FixedSizeInputStreamBuffer inBuffer(chunk.memory,chunk.size);
	
	/*if(chunk.size > 0)
     		free(chunk.memory);*/
	
	xdata::exdr::Serializer serializer;

	try 
	{
		serializer.import(/*&(*rtable)*/ rtable, &inBuffer );
		//serializer.import( rtable, &inBuffer );
	}
	catch(xdata::exception::Exception & e )
	{
		LOG4CPLUS_ERROR(this->getApplicationLogger(),"xdata::exdr::Serializer exception occured...");
		LOG4CPLUS_ERROR(this->getApplicationLogger(),xcept::stdformat_exception_history(e));
			
		if(chunk.size > 0)
     			free(chunk.memory);
			
		return -1 ;
	}
	
   	if(chunk.size > 0)
     		free(chunk.memory);
	
   	return 0;
}




/** Timer method called, this method inject the method dcuReadWorkLoop in the queue to be executed
 */
void xmas2dqm::wse::XmasToDQM::timeExpired (toolbox::task::TimerEvent& e) 
{
	//keep log how many times the period for parsing LAS expired...
	static int times = 0;
	
	times++;	
	
	LOG4CPLUS_INFO (getApplicationLogger(), "timeExpired was called... : time " << times <<"th");
	LASReadoutWorkLoop_->submit(LASReadout_);
 
}


void *xmas2dqm::wse::XmasToDQM::myrealloc(void *ptr, size_t size)
{
    	/* There might be a realloc() out there that doesn't like reallocing
       	NULL pointers, so we take care of it here */
    	if(ptr)
      		return realloc(ptr, size);
    	else
      		return malloc(size);
}


//This function gets called by libcurl as soon as there is data received that needs to be saved
size_t xmas2dqm::wse::XmasToDQM::WriteMemoryCallback(void *ptr, size_t size, size_t nmemb, void *data)
{
    	size_t realsize = size * nmemb;
    	struct MemoryStruct *mem = (struct MemoryStruct *)data;
  
   	mem->memory = (char *)myrealloc(mem->memory, mem->size + realsize + 1);
    	if (mem->memory) 
	{
      		memcpy(&(mem->memory[mem->size]), ptr, realsize);
      		mem->size += realsize;
      		mem->memory[mem->size] = 0;
    	}
    	return realsize;
}



//listening to exported paramater values
void xmas2dqm::wse::XmasToDQM::actionPerformed (xdata::Event& e) 
{ 
	LOG4CPLUS_INFO(getApplicationLogger(), "start of actionperformed");
	LOG4CPLUS_INFO(getApplicationLogger(), e.type());
 	
	// update exported parameters		
	if (e.type() == "ItemChangedEvent")
	{
		std::string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();	
		
		if ( item == "Period")
		{

			LOG4CPLUS_INFO(getApplicationLogger(), "item = " + item);		
			
			if(fsm_.getStateName (fsm_.getCurrentState()) != "Enabled")
			{
				return;
			}
		
			try
			{
				LASReadoutTimer_->remove(std::string("LASReadout"));
			}
			catch(toolbox::task::exception::NotActive &e)
			{
				LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::NotActive exception occured...");
			}
			catch(toolbox::task::exception::NoJobs)
			{
				LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::NoJobs exception occured...");
			}
			catch(toolbox::task::exception::JobNotFound &e)
			{
				LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::JobNotFound exception occured...");
				toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
        
			}
		
			//LASReadoutTimer_->stop();
			
			toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
        
			startLASReadout_ = toolbox::TimeVal::gettimeofday();
        
			LASReadoutTimer_->scheduleAtFixedRate( startLASReadout_, this, interval,  0, std::string("LASReadout") );	
		} 
	
	}
	
	LOG4CPLUS_INFO(getApplicationLogger(), "end of actionperformed");
}


// LASReadoutTimer_->scheduleAtFixedRate( startLASReadout_, this, interval,  0, std::string("") );	
// //! Schedules the specified task for execution after the specified delay
// 		void scheduleAtFixedRate (
// 					toolbox::TimeVal& start,
// 					toolbox::task::TimerListener* listener,
// 					toolbox::TimeInterval& period,
// 					void* context,
// 					const std::string& name) 
// 				throw (toolbox::task::exception::InvalidListener, toolbox::task::exception::InvalidSubmission, toolbox::task::exception::NotActive);



xmas2dqm::wse::XmasToDQM::~XmasToDQM()
{

}


xoap::MessageReference xmas2dqm::wse::XmasToDQM::fireEvent (xoap::MessageReference msg) 
	throw (xoap::exception::Exception)
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


			try 
			{
				toolbox::Event::Reference e(new toolbox::Event(commandName,this));
				fsm_.fireEvent(e);
			}
			catch (toolbox::fsm::exception::Exception & e)
			{
				XCEPT_RETHROW(xoap::exception::Exception, "invalid command", e);
			}

			xoap::MessageReference reply = xoap::createMessage();
			xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
			xoap::SOAPName responseName = envelope.createName( commandName +"Response", "xdaq", XDAQ_NS_URI);
			// xoap::SOAPBodyElement e = envelope.getBody().addBodyElement ( responseName );
			(void) envelope.getBody().addBodyElement ( responseName );
			return reply;
		}
	}

	XCEPT_RAISE(xcept::Exception,"command not found");		
}

	
xoap::MessageReference xmas2dqm::wse::XmasToDQM::reset (xoap::MessageReference msg) throw (xoap::exception::Exception)
{
	LOG4CPLUS_INFO (getApplicationLogger(), "New state before reset is: " << fsm_.getStateName (fsm_.getCurrentState()) );

	fsm_.reset();
	state_ = fsm_.getStateName (fsm_.getCurrentState());

	xoap::MessageReference reply = xoap::createMessage();
	xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
	xoap::SOAPName responseName = envelope.createName("ResetResponse", "xdaq", XDAQ_NS_URI);
	// xoap::SOAPBodyElement e = envelope.getBody().addBodyElement ( responseName );
	(void) envelope.getBody().addBodyElement ( responseName );

	LOG4CPLUS_INFO (getApplicationLogger(), "New state after reset is: " << fsm_.getStateName (fsm_.getCurrentState()) );

	return reply;
}

	
void xmas2dqm::wse::XmasToDQM::EnableAction (toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) 
{
	LOG4CPLUS_INFO (getApplicationLogger(), e->type());
	
	
	try
	{
		LASReadoutTimer_->remove(std::string("LASReadout"));
	}
	catch(toolbox::task::exception::NotActive &e)
	{
		LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::NotActive exception occured...");
	}
	catch(toolbox::task::exception::NoJobs)
	{
		LOG4CPLUS_INFO(getApplicationLogger(), "toolbox::task::exception::NoJobs exception occured...");
	}
	catch(toolbox::task::exception::JobNotFound &e)
	{
		LOG4CPLUS_INFO(getApplicationLogger(), "toolbox::task::exception::JobNotFound exception occured...");
		toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
        
	}
	
	
	toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
        
	startLASReadout_ = toolbox::TimeVal::gettimeofday();
        
	LASReadoutTimer_->scheduleAtFixedRate( startLASReadout_, this, interval,  0, std::string("LASReadout") );
	
}


void xmas2dqm::wse::XmasToDQM::HaltAction (toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception) 
{
	LOG4CPLUS_INFO (getApplicationLogger(), e->type());
	
	try
	{
		LASReadoutTimer_->remove(std::string("LASReadout"));
	}
	catch(toolbox::task::exception::NotActive &e)
	{
		LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::NotActive exception occured...");
	}
	catch(toolbox::task::exception::NoJobs)
	{
		LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::NoJobs exception occured...");
	}
	catch(toolbox::task::exception::JobNotFound &e)
	{
		LOG4CPLUS_WARN(getApplicationLogger(), "toolbox::task::exception::JobNotFound exception occured...");
		toolbox::TimeInterval interval(/*10*/atoi(Period_.toString().c_str()),0); // period of 8 secs 
        
	}

}



void xmas2dqm::wse::XmasToDQM::stateChanged (toolbox::fsm::FiniteStateMachine & fsm) throw (toolbox::fsm::exception::Exception)
{
	// Reflect the new state
	state_ = fsm.getStateName (fsm.getCurrentState());
	LOG4CPLUS_INFO (getApplicationLogger(), "New state is:" << fsm.getStateName (fsm.getCurrentState()) );
}

void xmas2dqm::wse::XmasToDQM::failedTransition (toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
	toolbox::fsm::FailedEvent & fe = dynamic_cast<toolbox::fsm::FailedEvent&>(*e);
	LOG4CPLUS_INFO (getApplicationLogger(), "Failure occurred when performing transition from: "  <<
			fe.getFromState() <<  " to: " << fe.getToState() << " exception: " << fe.getException().what() );
}


