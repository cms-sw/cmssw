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

  mdata.met = 0.;
  mdata.metx = 0.;
  mdata.mety = 0.;
  mdata.metz = 0.;
  mdata.sumet = 0.;
  mdata.phi = 0.;

}
