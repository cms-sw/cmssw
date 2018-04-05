// -*- C++ -*-
//
// Package:    FastSimulation/Calorimetry
// Class:      KKCorrectionFactors
//
/**\class KKCorrectionFactorsr
 Description: Returns scale from an TH3F histogram in a root file
*/
//
// Original Author:  Maximilian Knut Kiesel
//         Created:  Fri, 07 Aug 2015
//
//


#ifndef CALORESPONSE_H
#define CALORESPONSE_H

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TH3F.h>
#include <TFile.h>
#include <TROOT.h>


class KKCorrectionFactors {

 public:
  /* Constructor: pset must contain two strings: "fileName" is the name of the
   * file in which the TH3F named "histogramName" is saved.
   */
  KKCorrectionFactors( const edm::ParameterSet& pset );
  ~KKCorrectionFactors() { delete h3_; };

  float getScale( float genEnergy, float genEta, float simEnergy ) const;

 private:
  // histogram which contains the scales
  TH3F* h3_;
  bool interpolate3D_;

};

#endif
