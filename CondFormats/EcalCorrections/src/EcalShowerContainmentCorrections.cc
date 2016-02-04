// Implementation of class EcalShowerContainmentCorrections
// Author: Stefano Argiro'
// $Id: EcalShowerContainmentCorrections.cc,v 1.2 2007/07/16 17:30:54 meridian Exp $

#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrections.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
//#include <iostream>


const EcalShowerContainmentCorrections::Coefficients
EcalShowerContainmentCorrections::correctionCoefficients(const EBDetId& centerxtal) const 
{
  GroupMap::const_iterator iter = groupmap_.find(centerxtal.rawId());

  if (iter!=groupmap_.end()) {
    int group =iter->second;
    return coefficients_[group-1];
  }

  edm::LogError("ShowerContaiment Correction not found");
  return Coefficients();

}


void 
EcalShowerContainmentCorrections::fillCorrectionCoefficients(const EBDetId& xtal, int group,const Coefficients&  coefficients){

  // do not replace if we already have the xtal
  if (groupmap_.find(xtal)!=groupmap_.end()) return;
  groupmap_[xtal]=group;


  if (coefficients_.size()<(unsigned int)(group)) {
    coefficients_.resize(group);
    coefficients_[group-1]=coefficients;
  }

  // we don't need to fill coefficients if the group has already been inserted


}


void 
EcalShowerContainmentCorrections::fillCorrectionCoefficients(const int supermodule, const int module, 
					    const Coefficients&  coefficients){


  if (module>EBDetId::kModulesPerSM) {
    edm::LogError("Invalid Module Number");
    return;
  }


  // what is EBDetID::kModuleBoundaries ? we better redefine them here ...
  const int kModuleLow[]={1,501,901,1301};
  const int kModuleHigh[]={500,900,1300,1700};

  for (int xtal =kModuleLow[module-1] ; xtal <= kModuleHigh[module-1];++xtal){
    EBDetId detid(supermodule,xtal,EBDetId::SMCRYSTALMODE);
    fillCorrectionCoefficients(detid,module,coefficients);
  }
  
}

/** Calculate the correction for the given direction and type  */
const double 
EcalShowerContainmentCorrections::correctionXY(const EBDetId& xtal, 
				double position,  
			        EcalShowerContainmentCorrections::Direction dir,
				EcalShowerContainmentCorrections::Type type
                                ) const{
  
  GroupMap::const_iterator iter=groupmap_.find(xtal);
  if (iter==groupmap_.end()) return -1;

  int group=iter->second;
  EcalShowerContainmentCorrections::Coefficients coeff=coefficients_[group-1];

  int offset=0;
  
  if (dir==eY)    offset+=  2* Coefficients::kPolynomialDegree;
  if (position<0) offset+=     Coefficients::kPolynomialDegree;
  if (type==e5x5) offset+=  4* Coefficients::kPolynomialDegree;

  double corr=0;

  for (  int i=offset; 
	 i<offset+Coefficients::kPolynomialDegree; 
	 ++i){  
    corr+= coeff.data[i] * pow( position ,i-offset) ;    
  }

  return corr;  
}

const double 
EcalShowerContainmentCorrections::correction3x3(const EBDetId& xtal, 
					     const  math::XYZPoint& pos) const {
    
  double x= pos.X()*10; // correction functions use mm
  double y= pos.Y()*10;

  double corrx = correctionXY(xtal,x,eX,e3x3);
  double corry = correctionXY(xtal,y,eY,e3x3);
  
  return corrx*corry;
}




const double 
EcalShowerContainmentCorrections::correction5x5(const EBDetId& xtal, 
					     const  math::XYZPoint& pos) const {
    
  double x= pos.X()*10; // correction functions use mm
  double y= pos.Y()*10;

  double corrx = correctionXY(xtal,x,eX,e5x5);
  double corry = correctionXY(xtal,y,eY,e5x5);
  
  return corrx*corry;
}

