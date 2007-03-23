// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef ECALUNPACKEREXCEPTION_H
#define ECALUNPACKEREXCEPTION_H

#include <iostream>
#include <string>
#include <sstream>

using namespace std;


class ECALUnpackerException { 
  public :
		
    /**
     * Constructor
     */
    ECALUnpackerException(ostringstream a){ info_=a.str(); }
  
    ECALUnpackerException(string a){info_=a;}		
    /**
     * Exception's discription
     */
    string what() const throw() { return info_;}
			
  protected :
	
    string info_;

};

#endif
