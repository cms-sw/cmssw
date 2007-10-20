// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef ECALTBPARSERBLOCKEXCEPTION_H
#define ECALTBPARSERBLOCKEXCEPTION_H

#include <iostream>
#include <string>



class ECALTBParserBlockException{ 
		public :
		
			/**
			 * Constructor
			 */
  ECALTBParserBlockException(std::string exceptionInfo_){info_ = exceptionInfo_; }
		
		
			/**
			 * Exception's discription
			 */
			const char * what() const throw() {	return info_.c_str();}
			
		protected :
	
			std::string info_;

};

#endif
