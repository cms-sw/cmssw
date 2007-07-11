
//FWCore
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

//CalibFormats
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

//DataFormats
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

using namespace sistrip;

template <class T>
uint32_t SiStripRegionCabling::regions(edm::SiStripRefGetter<T>& getter, 
				       edm::Handle<edm::SiStripLazyGetter<T> > handle,
				       Position position, 
				       double deltaeta, 
				       double deltaphi) const {
  //Counter
  uint32_t count = 0;
  
  //Calculate rectangle of ineterest boundary
  PositionIndex index = positionIndex(position);
  uint32_t deta = (uint32_t)(deltaeta/regionDimensions().first);
  uint32_t dphi = (uint32_t)(deltaphi/regionDimensions().second);
  
  //Loop eta index
  for (uint32_t ieta = 0; ieta < 2*deta + 1; ieta++) {
    int etatemp = index.first - deta + ieta;
    if ((etatemp < 0) || (etatemp >= (int)etadivisions_)) continue;
    
    //Loop phi index
    for (uint32_t iphi = 0; iphi < 2*dphi + 1; iphi++) {
      int phitemp = index.second - dphi + iphi;
      if (phitemp >= (int)phidivisions_) phitemp -= phidivisions_;
      else if (phitemp < 0) phitemp += phidivisions_;
      
      //Update SiStripRefGetter<T>
      getter.push_back(region(PositionIndex((uint32_t)etatemp,(uint32_t)phitemp)),
		       handle);
      count++;
    }
  }

  //Return counter
  return count;
}

template <class T>
uint32_t SiStripRegionCabling::regions(edm::SiStripRefGetter<T>& getter, 
				       edm::Handle<edm::SiStripLazyGetter<T> > handle, 
				       Position position, 
				       double dR) const {
  return regions<T>(getter, handle, position, 1./sqrt(2)*dR*dR,1./sqrt(2)*dR*dR);
}

template <class T>
uint32_t SiStripRegionCabling::elements(edm::SiStripRefGetter<T>& getter,
					edm::Handle<edm::SiStripLazyGetter<T> > handle, 
					Position position, 
					double deltaeta, 
					double deltaphi, 
					SubDet subdet, 
					Layer layer) const{
  
  //Counter
  uint32_t count = 0;  
  
  //Calculate rectangle of ineterest boundary
  PositionIndex index = positionIndex(position);
  uint32_t deta = (uint32_t)(deltaeta/regionDimensions().first);
  uint32_t dphi = (uint32_t)(deltaphi/regionDimensions().second);
  
  //Loop eta index
  for (uint32_t ieta = 0; ieta < 2*deta + 1; ieta++) {
    int etatemp = index.first - deta + ieta;
    if ((etatemp < 0) || (etatemp >= (int)etadivisions_)) continue;
    
    //Loop phi index
    for (uint32_t iphi = 0; iphi < 2*dphi + 1; iphi++) {
      int phitemp = index.second - dphi + iphi;
      if (phitemp >= (int)phidivisions_) phitemp -= phidivisions_;
      else if (phitemp < 0) phitemp += phidivisions_;
      
      //Update SiStripRefGetter<T>
      getter.push_back(elementIndex(region(PositionIndex((uint32_t)etatemp,(uint32_t)phitemp)),subdet,layer),
		       handle);
      count++;
    }
  }

  //Return counter
  return count;
}

const SiStripRegionCabling::PositionIndex SiStripRegionCabling::positionIndex(Position position) const {
  uint32_t eta = (uint32_t)((position.first+etamax_)*etadivisions_/(2.*etamax_));
  if (eta >= etadivisions_) eta = etadivisions_ - 1;
  if (eta < 0) eta = 0;
  return PositionIndex(eta,(uint32_t)((position.second+M_PI)*phidivisions_/(2.*M_PI)));
}

const SiStripRegionCabling::Region SiStripRegionCabling::region(Position position) const {
  PositionIndex index = positionIndex(position); 
  return region(index);
}

const SiStripRegionCabling::SubDet SiStripRegionCabling::subdetFromDetId(uint32_t detid) {

  SiStripDetId::SubDetector subdet = SiStripDetId(detid).subDetector();
  if (subdet == 3) return SiStripRegionCabling::TIB;
  else if (subdet == 5) return SiStripRegionCabling::TOB;
  else if (subdet == 4) return SiStripRegionCabling::TID;
  else if (subdet == 6) return SiStripRegionCabling::TEC;
  else return SiStripRegionCabling::UNKNOWN;
}

const SiStripRegionCabling::Layer SiStripRegionCabling::layerFromDetId(uint32_t detid) {
 
  if (subdet(detid) == SiStripRegionCabling::TIB) return TIBDetId(detid).layer();
  else if (subdet(detid) == SiStripRegionCabling::TOB) return TOBDetId(detid).layer(); 
  else if (subdet(detid) == SiStripRegionCabling::TEC) return TECDetId(detid).wheel();
  else if (subdet(detid) == SiStripRegionCabling::TID) return TIDDetId(detid).wheel();
  else return 0;
}

EVENTSETUP_DATA_REG(SiStripRegionCabling);
