//-------------------------------------------------
//
//   \class L1MuGMTLUTHelpers
/**
 *   Description: Helper Functions for string handling in L1MuGMTLUT
 *                (could be found in Cobra but classes
 *                 should be independet of COBRA/ORCA)
 *
*/ 
//
//   $Date: 2003/12/18 19:24:07 $
//   $Revision: 1.1 $
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
using namespace std;

class L1MuGMTLUTHelpers {

  public:  
    /// constructor
    L1MuGMTLUTHelpers() {};

    /// destructor
    virtual ~L1MuGMTLUTHelpers() {};

    /// Lookup Functions

    /// some string tools

    class Tokenizer : public std::vector<string> {
    public:
      Tokenizer(const string & sep, const string & input){
	size_type i=0, j=0;
	while( (j=input.find(sep,i))!=string::npos) {
	  push_back(input.substr(i,j-i));
	  i = j+sep.size();
	}
	push_back(input.substr(i));
      }; 

    };

    static int replace(string& input, const string& gone, const string& it, bool multiple);

    static string upperCase(const string& s);

    static string lowerCase(const string& s);

};

#endif












