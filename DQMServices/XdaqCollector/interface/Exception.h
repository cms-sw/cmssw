// $Id: Exception.h,v 1.1 2008/07/01 13:22:26 ameyer Exp $

/*************************************************************************
 * XDAQ Components for Distributed Data Acquisition                      *
 * Copyright (C) 2000-2004, CERN.			                 *
 * All rights reserved.                                                  *
 * Authors: J. Gutleber and L. Orsini					 *
 *                                                                       *
 * For the licensing terms see LICENSE.		                         *
 * For the list of contributors see CREDITS.   			         *
 *************************************************************************/

#ifndef _xmas2dqm_wse_exception_Exception_h_
#define _xmas2dqm_wse_exception_Exception_h_

#include "xmas/exception/Exception.h"


namespace xmas2dqm {
	namespace wse {
		namespace exception { 
			class Exception: public xmas::exception::Exception 
			{
				public: 
				Exception( std::string name, std::string message, std::string module, int line, std::string function ): 
					xmas::exception::Exception(name, message, module, line, function) 
				{} 

				Exception( std::string name, std::string message, std::string module, int line, std::string function,
					xcept::Exception& e ): 
					xmas::exception::Exception(name, message, module, line, function, e) 
				{} 

			};  
		}
	}		
}

#endif
