//-------------------------------------------------
//
//   \class L1MuGMTLUTHelpers
/**
 *   Description: Helper Functions for std::string handling in L1MuGMTLUT
 *                (could be found in Cobra but classes
 *                 should be independet of COBRA/ORCA)
 *
*/ 
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.2 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTLUTHelpers_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTLUTHelpers_h

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTLUTHelpers {

  public:  
    /// constructor
    L1MuGMTLUTHelpers() {};

    /// destructor
    virtual ~L1MuGMTLUTHelpers() {};

    /// Lookup Functions

    /// some std::string tools

    class Tokenizer : public std::vector<std::string> {
    public:
      Tokenizer(const std::string & sep, const std::string & input){
	size_type i=0, j=0;
	while( (j=input.find(sep,i))!=std::string::npos) {
	  push_back(input.substr(i,j-i));
	  i = j+sep.size();
	}
	push_back(input.substr(i));
      }; 

    };

    static int replace(std::string& input, const std::string& gone, const std::string& it, bool multiple);

    static std::string upperCase(const std::string& s);

    static std::string lowerCase(const std::string& s);

};

#endif












