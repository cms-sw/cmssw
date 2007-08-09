
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

SiStripRegionCabling::SiStripRegionCabling(const uint32_t etadivisions, const uint32_t phidivisions, const double etamax) :

  etadivisions_(static_cast<int>(etadivisions)),
  phidivisions_(static_cast<int>(phidivisions)),
  etamax_(etamax),
  regioncabling_()

{}

const SiStripRegionCabling::PositionIndex SiStripRegionCabling::positionIndex(const Position position) const {
  int eta = (etamax_) ? static_cast<int>((position.first+etamax_)*etadivisions_/(2.*etamax_)) : 0;
  int phi = static_cast<int>((position.second+M_PI)*phidivisions_/(2.*M_PI));
  if (eta > etadivisions_-1) eta = etadivisions_-1;
  else if (eta < 0) eta = 0;
  return PositionIndex(static_cast<uint32_t>(eta),static_cast<uint32_t>(phi));
}

const SiStripRegionCabling::Region SiStripRegionCabling::region(const Position position) const {
  PositionIndex index = positionIndex(position); 
  return region(index);
}

void SiStripRegionCabling::increment(PositionIndex& index, int deta, int dphi) const {
  
  int eta = static_cast<int>(index.first);
  eta+=deta;
  if (eta > etadivisions_-1) eta = etadivisions_-1;
  else if (eta < 0) eta = 0;
  index.first = static_cast<uint32_t>(eta);
  
  int phi = static_cast<int>(index.second);
  phi+=dphi;
  while (phi<0) phi+=phidivisions_;
  while (phi>phidivisions_-1) phi-=phidivisions_;
  index.second = static_cast<uint32_t>(phi);
}

const SiStripRegionCabling::SubDet SiStripRegionCabling::subdetFromDetId(const uint32_t detid) {

  SiStripDetId::SubDetector subdet = SiStripDetId(detid).subDetector();
  if (subdet == 3) return SiStripRegionCabling::TIB;
  else if (subdet == 5) return SiStripRegionCabling::TOB;
  else if (subdet == 4) return SiStripRegionCabling::TID;
  else if (subdet == 6) return SiStripRegionCabling::TEC;
  else return SiStripRegionCabling::UNKNOWN;
}

const SiStripRegionCabling::Layer SiStripRegionCabling::layerFromDetId(const uint32_t detid) {
 
  if (subdet(detid) == SiStripRegionCabling::TIB) return TIBDetId(detid).layer();
  else if (subdet(detid) == SiStripRegionCabling::TOB) return TOBDetId(detid).layer(); 
  else if (subdet(detid) == SiStripRegionCabling::TEC) return TECDetId(detid).wheel();
  else if (subdet(detid) == SiStripRegionCabling::TID) return TIDDetId(detid).wheel();
  else return 0;
}

EVENTSETUP_DATA_REG(SiStripRegionCabling);
