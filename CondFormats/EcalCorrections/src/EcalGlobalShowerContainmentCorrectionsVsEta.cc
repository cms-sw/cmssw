// Implementation of class EcalGlobalShowerContainmentCorrectionsVsEta
// Author: Paolo Meridiani
// $Id: $

#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
//#include <iostream>


const EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients
EcalGlobalShowerContainmentCorrectionsVsEta::correctionCoefficients() const {
  return coefficients_;
}


void 
EcalGlobalShowerContainmentCorrectionsVsEta::fillCorrectionCoefficients(const Coefficients&  coefficients)
{
  coefficients_=coefficients;
}

/** Calculate the correction for the given direction and type  */
const double 
EcalGlobalShowerContainmentCorrectionsVsEta::correction(const DetId& xtal, 
							  EcalGlobalShowerContainmentCorrectionsVsEta::Type type
							  ) const
{

  int offset=0;
  
  if (type==e5x5) offset+=  4* Coefficients::kCoefficients;
  
  double corr=0;

  for (  int i=offset; 
	 i<offset+Coefficients::kCoefficients; 
	 ++i){  
    //FIXME implement the correct function 
    corr+= coefficients_.data[i] * pow( EBDetId(xtal).ieta(), i-offset) ;    
  }

  return corr;  
}

const double 
EcalGlobalShowerContainmentCorrectionsVsEta::correction3x3(const DetId& xtal) const 
{
  double corr = correction(xtal,e3x3);
  return corr;
}




const double 
EcalGlobalShowerContainmentCorrectionsVsEta::correction5x5(const DetId& xtal) const 
{ 
  double corr = correction(xtal,e5x5);
  return corr;
}

