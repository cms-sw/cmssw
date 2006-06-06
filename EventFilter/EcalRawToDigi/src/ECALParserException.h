// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef ECALPARSEREXCEPTION_H
#define ECALPARSEREXCEPTION_H

#include <iostream>
#include <string>

using std::string;


class ECALParserException { 
		public :
		
			/**
			 * Constructor
			 */
			ECALParserException(string exceptionInfo){info_ = exceptionInfo; }
		
		
			/**
			 * Exception's discription
			 */
			const char * what() const throw() {	return info_.c_str();}
			
		protected :
	
			string info_;

};

#endif
