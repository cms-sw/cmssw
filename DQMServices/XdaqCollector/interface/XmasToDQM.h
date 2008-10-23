// $Id: XmasToDQM.h,v 1.1 2008/07/01 13:22:26 ameyer Exp $

/*************************************************************************
 * XDAQ Components for Distributed Data Acquisition                      *
 * Copyright (C) 2000-2004, CERN.			                 *
 * All rights reserved.                                                  *
 * Authors: J. Gutleber and L. Orsini					 *
 *                                                                       *
 * For the licensing terms see LICENSE.		                         *
 * For the list of contributors see CREDITS.   			         *
 *************************************************************************/

#ifndef _parse_xmas2dqm_wse_Application_h_
#define _parse_xmas2dqm_wse_Application_h_

#include <string>
#include <map>

#include "toolbox/ActionListener.h"
#include "toolbox/task/AsynchronousEventDispatcher.h"

#include "xdaq/ApplicationDescriptorImpl.h"
#include "xdaq/Application.h" 
#include "xdaq/ApplicationContext.h" 

#include "xdata/String.h"
#include "xdata/Vector.h"
#include "xdata/Boolean.h"
#include "xdata/ActionListener.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"
#include "xoap/Method.h"
#include "xdata/UnsignedInteger64.h"

#include "ws/addressing/EndpointReference.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"
#include "xgi/exception/Exception.h"
#include "Exception.h"
#include "ToDqm.h"

#include "xdaq/ContextTable.h"

#include "xdaq/NamespaceURI.h"

#include "toolbox/fsm/FiniteStateMachine.h"
#include "toolbox/fsm/FailedEvent.h"


// for the work loop and timer
#include "toolbox/task/WorkLoopFactory.h"
#include "toolbox/task/WaitingWorkLoop.h" 
#include "toolbox/task/Timer.h"
#include "toolbox/task/TimerFactory.h"

#include <curl/curl.h>
#include <curl/types.h>
#include <curl/easy.h>
#include <fstream> // for ifstream, ofstream, ios_base
#include <iostream>

//#include "DQMServices/XdaqCollector/interface/FlashlistElements.h"

namespace xmas2dqm 
{
	namespace wse 
	{
		struct MemoryStruct 
		{		
			char *memory;
   			size_t size;
		};
	
		class XmasToDQM :public xdaq::Application, public toolbox::task::TimerListener, public xdata::ActionListener
			/*,public toolbox::ActionListener,*/ 
		{
			public:

			XDAQ_INSTANTIATOR();

			XmasToDQM(xdaq::ApplicationStub* s) throw (xdaq::exception::Exception);
			~XmasToDQM();

			// Callback for listening to exported parameter values
			void actionPerformed (xdata::Event& e);
			
			//void actionPerformed( toolbox::Event& event );

			//
			// XGI Interface
			//
			void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
			

			//
			// SOAP interface
			//
			
			//! Receive metrics from a sensor
			xoap::MessageReference report (xoap::MessageReference msg) throw (xoap::exception::Exception);
				
			protected:
			
					
			
			// SOAP Callback trigger state change 
			//
			xoap::MessageReference fireEvent (xoap::MessageReference msg) 
				throw (xoap::exception::Exception);
				
			// SOAP Callback to reset the state machine
			//
			xoap::MessageReference reset (xoap::MessageReference msg) 
				throw (xoap::exception::Exception);
			
			// Finite State Machine action callbacks
			//
			void ConfigureAction (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);
	
			void EnableAction (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);
	
			void SuspendAction (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);
	
			void ResumeAction (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);

			void HaltAction (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);
	
			void failedTransition (toolbox::Event::Reference e) 
				throw (toolbox::fsm::exception::Exception);
	
			// Finite State Machine callback for entring state
			//
			void stateChanged (toolbox::fsm::FiniteStateMachine & fsm) 
				throw (toolbox::fsm::exception::Exception);
				
			bool LASReadoutWorkLoop (toolbox::task::WorkLoop* wl);
			
			void timeExpired (toolbox::task::TimerEvent& e); 
			
			static void *myrealloc(void *ptr, size_t size);
			
			static size_t WriteMemoryCallback(void *ptr, size_t size, size_t nmemb, void *data);
			
			int getEXDR_LAS(/*xdata::Table::Reference &*/xdata::Table *rtable);
			
			xdata::String state_; // used to reflect the current state to the outside world
			xdata::String LASurl_; //holds the value of the LAS URL 
			xdata::String Period_; //LAS parsing period time in seconds 
			xdata::String LASQueueSize_; //LAS parsing period time in seconds 
			
			//xdata::Bag<xmas2dqm::wse::FlashlistElements> flashlistMonitor_;
			
			
			private:

			
			// dqm hook
			xmas2dqm::wse::ToDqm *dqmHook_;
			
			//toolbox::task::AsynchronousEventDispatcher dispatcher_;
			//xdata::UnsignedInteger64T reportLostCounter_;
		
	
			toolbox::fsm::FiniteStateMachine fsm_; 
			
			//Working loop in the system
  			toolbox::task::WorkLoop* LASReadoutWorkLoop_ ;
			
			//method to be activated by the work loop
   			toolbox::task::ActionSignature* LASReadout_ ;
			
			toolbox::task::Timer * LASReadoutTimer_;
			toolbox::TimeVal startLASReadout_;
						
						
		};
	}
}
#endif
