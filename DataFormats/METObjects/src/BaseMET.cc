// File: BaseMET.cc
// Description: see BaseMET.h
// Author: Michael Schmitt, R. Cavanaugh University of Florida
// Creation Date:  MHS MAY 30, 2005 initial version

#include "DataFormats/METObjects/interface/BaseMET.h"

using namespace std;

BaseMET::BaseMET() 
{
  clearMET();
}

void BaseMET::clearMET() 
{
  data.label[0] = '\0';
  data.met   = 0.0;
  data.mex   = 0.0;
  data.mey   = 0.0;
  data.mez   = 0.0;
  data.sumet = 0.0;
  data.phi   = 0.0;
}
