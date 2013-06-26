//-------------------------------------------------
//
//   Class: L1MuGMTLUTHelpers
/**
 *   Description: String handling helper functions for L1MuGMTLUT
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLUTHelpers.h"
//---------------
// C++ Headers --
//---------------
#include <ctype.h>

using namespace std;

//--------------------------------------------------------------------------------
// Replace substring in string

int L1MuGMTLUTHelpers::replace(string & input, const string& gone, const string& it, bool multiple) {
  int n=0;
  size_t i = input.find(gone,0);
  while(i!=string::npos) {
      n++;
      input.replace(i,gone.size(),it);
      i = input.find(gone,i+(multiple ? 0 : it.size()));
  }
  return n;
}

//--------------------------------------------------------------------------------
// Make an uppercase copy of s

string L1MuGMTLUTHelpers::upperCase(const string& s) {
  char* buf = new char[s.length()];
  s.copy(buf, s.length());
  for(unsigned i = 0; i < s.length(); i++)
    buf[i] = toupper(buf[i]);
  string r(buf, s.length());
  delete buf;
  return r;
}



//--------------------------------------------------------------------------------
// Make an lowercase copy of s

string L1MuGMTLUTHelpers::lowerCase(const string& s) {
  char* buf = new char[s.length()];
  s.copy(buf, s.length());
  for(unsigned i = 0; i < s.length(); i++)
    buf[i] = tolower(buf[i]);
  string r(buf, s.length());
  delete buf;
  return r;
}





