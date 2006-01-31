// File: BaseMET.cc
// Description: see BaseMET.h
// Author: Michael Schmitt
// Creation Date:  MHS MAY 30, 2005 initial version

#include "DataFormats/METObjects/interface/BaseMET.h"

using namespace std;

BaseMET::BaseMET() {

  clearMET();

}

void BaseMET::clearMET() {

  data.met = 0.;
  data.metx = 0.;
  data.mety = 0.;
  data.metz = 0.;
  data.sumet = 0.;
  data.phi = 0.;

}
