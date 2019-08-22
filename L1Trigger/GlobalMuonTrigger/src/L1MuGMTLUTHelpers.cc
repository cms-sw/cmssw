//-------------------------------------------------
//
//   Class: L1MuGMTLUTHelpers
/**
 *   Description: String handling helper functions for L1MuGMTLUT
 * 
*/
//
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
#include <cctype>

using namespace std;

//--------------------------------------------------------------------------------
// Replace substring in string

int L1MuGMTLUTHelpers::replace(string& input, const string& gone, const string& it, bool multiple) {
  int n = 0;
  size_t i = input.find(gone, 0);
  while (i != string::npos) {
    n++;
    input.replace(i, gone.size(), it);
    i = input.find(gone, i + (multiple ? 0 : it.size()));
  }
  return n;
}
