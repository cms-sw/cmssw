// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef ECALUNPACKEREXCEPTION_H
#define ECALUNPACKEREXCEPTION_H

#include <iostream>
#include <string>
#include <sstream>


class ECALUnpackerException { 
  public :
		
    /**
     * Constructor
     */
    ECALUnpackerException(std::ostringstream a){ info_=a.str(); }
  
    ECALUnpackerException(std::string a){info_=a;}		
    /**
     * Exception's discription
     */
    std::string what() const throw() { return info_;}
			
  protected :
	
    std::string info_;

};

#endif
