// $Id: ToDqm.h,v 1.2 2008/10/13 13:03:50 vpatras Exp $

/*************************************************************************
 * XDAQ Components for Distributed Data Acquisition                      *
 * Copyright (C) 2000-2004, CERN.			                 *
 * All rights reserved.                                                  *
 * Authors: J. Gutleber and L. Orsini					 *
 *                                                                       *
 * For the licensing terms see LICENSE.		                         *
 * For the list of contributors see CREDITS.   			         *
 *************************************************************************/

#ifndef _xmas2dqm_wse_ToDqm_h_
#define _xmas2dqm_wse_ToDqm_h_

#include <string>
#include <map>
#include "xdaq/ApplicationDescriptor.h"
#include "Exception.h"

#include "xdata/String.h"
#include "xdata/Vector.h"
#include "xdata/Boolean.h"
#include "xdata/ActionListener.h"
#include "xdata/UnsignedInteger64.h"
#include "xdata/Table.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <queue>
#include "toolbox/BSem.h"
#include <pthread.h>

/*#ifndef Qsize_max
#define Qsize_max 1000
#endif*/

#include "DQMServices/XdaqCollector/interface/FlashlistElements.h"

namespace xmas2dqm
{
	namespace wse
	{
		class ToDqm 
		{
			public:
		
			void digest(
					const std::string& flashListName, 
					const std::string& originator, 
					const std::string& tag, 
					/*xdata::Table::Reference table*/
					xdata::Table *table
					)
					throw (xmas2dqm::wse::exception::Exception );
					
			void free_memory();
			
			xdata::String runNumber_;
			
			std::queue<xdata::Table::Reference> QTable_;
			
			std::queue<xdata::Table *> MemoryTable_;
			
			xdata::Bag<xmas2dqm::wse::FlashlistElements> flashlistMonitor_;
			
			//semaphore to protect access to runNumber_
			toolbox::BSem BSem_;		
			
			//allows syncronized access to the queue of LAS data
			pthread_mutex_t LASmutex_;
	
                        //represents if the queue is (q.size reached max size)	
			pthread_cond_t more_;
			
			//represents if there is space in queue (q.size less than max size)
			pthread_cond_t less_;
			
					
			static ToDqm *instance();
		private:
			ToDqm();
			~ToDqm();
			static ToDqm* instance_;
			//DQMStore *be_;
			//MonitorElement *me_;
			int messageCount_;
				
			
		};
	
	}
}	

#endif
