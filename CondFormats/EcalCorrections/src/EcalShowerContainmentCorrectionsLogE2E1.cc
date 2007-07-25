// Implementation of class EcalShowerContainmentCorrectionsLogE2E1
// Author: Stefano Argiro'
// $Id: EcalShowerContainmentCorrectionsLogE2E1.cc,v 1.1 2007/05/15 20:37:22 argiro Exp $

#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrectionsLogE2E1.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
//#include <iostream>


const EcalShowerContainmentCorrectionsLogE2E1::Coefficients
EcalShowerContainmentCorrectionsLogE2E1::correctionCoefficients(const EBDetId& centerxtal) const {

  GroupMap::const_iterator iter = groupmap_.find(centerxtal.rawId());

  if (iter!=groupmap_.end()) {
    int group =iter->second;
    return coefficients_[group-1];
  }

  edm::LogError("ShowerContaiment Correction not found");
  return Coefficients();

}


void 
EcalShowerContainmentCorrectionsLogE2E1::fillCorrectionCoefficients(const EBDetId& xtal, int group,const Coefficients&  coefficients){

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
EcalShowerContainmentCorrectionsLogE2E1::fillCorrectionCoefficients(const int supermodule, const int module, 
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
EcalShowerContainmentCorrectionsLogE2E1::correctionEtaPhi(const EBDetId& xtal, 
				double position,  
			        EcalShowerContainmentCorrectionsLogE2E1::Direction dir,
				EcalShowerContainmentCorrectionsLogE2E1::Type type
                                ) const{
  
  GroupMap::const_iterator iter=groupmap_.find(xtal);
  if (iter==groupmap_.end()) return -1;

  int group=iter->second;
  EcalShowerContainmentCorrectionsLogE2E1::Coefficients coeff=coefficients_[group-1];

  int offset=0;
  
  if (dir==eEta)    offset+=  2* Coefficients::kPolynomialDegree;
  if (position<0)   offset+=     Coefficients::kPolynomialDegree;
  if (type==e5x5)   offset+=  4* Coefficients::kPolynomialDegree;

  double corr=0;

  for (  int i=offset; 
	 i<offset+Coefficients::kPolynomialDegree; 
	 ++i){  
    corr+= coeff.data[i] * pow( position ,i-offset) ;    
  }

  return corr;  
}

const double 
EcalShowerContainmentCorrectionsLogE2E1::correction3x3(const EBDetId& xtal, 
					      const double& loge2e1_eta, 
                                              const double& loge2e1_phi) const {

  double corrx = correctionEtaPhi(xtal,loge2e1_eta,eEta,e3x3);
  double corry = correctionEtaPhi(xtal,loge2e1_phi,ePhi,e3x3);
  
  return corrx*corry;
}




const double 
EcalShowerContainmentCorrectionsLogE2E1::correction5x5(const EBDetId& xtal, 
					      const double& loge2e1_eta, 
                                              const double& loge2e1_phi) const {
    
  double corrx = correctionEtaPhi(xtal,loge2e1_eta,eEta,e5x5);
  double corry = correctionEtaPhi(xtal,loge2e1_phi,ePhi,e5x5);
  
  return corrx*corry;
}

