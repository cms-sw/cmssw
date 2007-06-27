
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

SiStripRegionCabling::SiStripRegionCabling(const uint32_t EtaDivisions,const uint32_t PhiDivisions, const double EtaMax) :

  etadivisions_(EtaDivisions),
  phidivisions_(PhiDivisions),
  etamax_(EtaMax),
  regioncabling_()

{;}

const SiStripRegionCabling::Regions SiStripRegionCabling::regions(Position position, double dR) const {
  Regions regions;
  regions.reserve(etadivisions_*phidivisions_); 
  
  PositionIndex index = positionIndex(position);

  uint32_t deta = (uint32_t)(dR/regionDimensions().first);
  uint32_t dphi = (uint32_t)(dR/regionDimensions().second);

  for (uint32_t ieta = 0; ieta < 2*deta + 1; ieta++) {
    for (uint32_t iphi = 0; iphi < 2*dphi + 1; iphi++) {
      int etatemp = index.first - deta + ieta;
      int phitemp = index.second - dphi + iphi;
      if ((etatemp < 0) || (etatemp >= (int)etadivisions_)) continue;
      if (phitemp >= (int)phidivisions_) phitemp -= phidivisions_;
      else if (phitemp < 0) phitemp += phidivisions_;
      regions.push_back(region(PositionIndex((uint32_t)etatemp,(uint32_t)phitemp)));
    }
  }
  return regions;
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
