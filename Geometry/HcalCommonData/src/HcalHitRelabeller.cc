#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

HcalHitRelabeller::HcalHitRelabeller(bool nd) : 
  theRecNumber(nullptr),
  neutralDensity_(nd) { }

void HcalHitRelabeller::process(std::vector<PCaloHit>& hcalHits) {

  if (theRecNumber) {
    for (auto & hcalHit : hcalHits) {

#ifdef EDM_ML_DEBUG
      std::cout << "Hit[" << ii << "] " << std::hex << hcalHits[ii].id() 
		<< std::dec << " Neutral density " << neutralDensity_ << "\n";
#endif
      double energy = (hcalHit.energy());
      if (neutralDensity_) {
	energy *= (energyWt(hcalHit.id()));
	hcalHit.setEnergy(energy);
      }
      DetId newid = relabel(hcalHit.id());
#ifdef EDM_ML_DEBUG
      std::cout << "Hit " << ii << " out of " << hcalHits.size() << " " 
		<< std::hex << newid.rawId() << std::dec << " E " << energy 
                << std::endl;
#endif
      hcalHit.setID(newid.rawId());
#ifdef EDM_ML_DEBUG
      std::cout << "Modified Hit " << HcalDetId(hcalHits[ii].id()) <<std::endl;
#endif
    }
  } else {
    edm::LogWarning("HcalSim") << "HcalHitRelabeller: no valid HcalDDDRecConstants";
  }
  
}

void HcalHitRelabeller::setGeometry(const HcalDDDRecConstants *& recNum) {
  theRecNumber = recNum;
}

DetId HcalHitRelabeller::relabel(const uint32_t testId) const {
  return HcalHitRelabeller::relabel(testId, theRecNumber);
}

DetId HcalHitRelabeller::relabel(const uint32_t testId, const HcalDDDRecConstants * theRecNumber) {

#ifdef EDM_ML_DEBUG
  std::cout << "Enter HcalHitRelabeller::relabel " << std::endl;
#endif
  HcalDetId hid;
  int       det, z, depth, eta, phi, layer, sign;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
#ifdef EDM_ML_DEBUG
  std::cout << "det: " << det << " "
  	    << "z: " << z << " "
   	    << "depth: " << depth << " "
   	    << "ieta: " << eta << " "
   	    << "iphi: " << phi << " "
   	    << "layer: " << layer << std::endl;
#endif
  sign=(z==0)?(-1):(1);
  HcalDDDRecConstants::HcalID id = theRecNumber->getHCID(det,sign*eta,phi,layer,depth);

  if (id.subdet==int(HcalBarrel)) {
    hid=HcalDetId(HcalBarrel,sign*id.eta,id.phi,id.depth);        
  } else if (id.subdet==int(HcalEndcap)) {
    hid=HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalOuter)) {
    hid=HcalDetId(HcalOuter,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalForward)) {
    hid=HcalDetId(HcalForward,sign*id.eta,id.phi,id.depth);
  }
#ifdef EDM_ML_DEBUG
  std::cout << " new HcalDetId -> hex.RawID = "
	    << std::hex << hid.rawId() << std::dec;
  std::cout.flush();
  std::cout << " det, z, depth, eta, phi = " << det << " "
	    << z << " "<< id.depth << " " << id.eta << " "
	    << id.phi << " ---> " << hid << std::endl;  
#endif
  return hid;
}

double HcalHitRelabeller::energyWt(const uint32_t testId) const {

  HcalDetId hid;
  int       det, z, depth, eta, phi, layer;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
  int       zside = (z==0) ? (-1) : (1);
  double    wt    = (((det==1) || (det==2)) && (depth == 1)) ? 
    theRecNumber->getLayer0Wt(det,phi,zside) : 1.0;
#ifdef EDM_ML_DEBUG
  std::cout << "EnergyWT::det: " << det << " z: " << z  << ":" << zside
            << " depth: " << depth << " ieta: " << eta << " iphi: " << phi
            << " layer: " << layer << " wt " << wt << std::endl;
#endif
  return wt;
}
