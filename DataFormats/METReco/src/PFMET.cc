// File: PFMET.cc
// Description: see PFMET.h
// Author: R. Remington
// Creation Date:  Oct. 2009

#include "DataFormats/METReco/interface/PFMET.h"

using namespace std;
using namespace reco;

//---------------------------------------------------------------------------
// Default Constructor;
//-----------------------------------
PFMET::PFMET()
{
  // Initialize the container
  pf_data.NeutralEMFraction = 0.0;
  pf_data.NeutralHadFraction = 0.0;
  pf_data.ChargedEMFraction = 0.0;
  pf_data.ChargedHadFraction = 0.0;
  pf_data.MuonFraction = 0.0;
  pf_data.Type6Fraction = 0.0;
  pf_data.Type7Fraction = 0.0;



}

