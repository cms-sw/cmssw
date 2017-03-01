#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Utilities/interface/typelookup.h"

using namespace sistrip;

SiStripRegionCabling::SiStripRegionCabling(const uint32_t etadivisions, const uint32_t phidivisions, const double etamax) :

  etadivisions_(static_cast<int>(etadivisions)),
  phidivisions_(static_cast<int>(phidivisions)),
  etamax_(etamax),
  regioncabling_()

{}

const SiStripRegionCabling::PositionIndex SiStripRegionCabling::positionIndex(const Position position) const {
  int eta = static_cast<int>((position.first+etamax_)*etadivisions_/(2.*etamax_));
  int phi = static_cast<int>((position.second+M_PI)*phidivisions_/(2.*M_PI));
  return PositionIndex(static_cast<uint32_t>(eta),static_cast<uint32_t>(phi));
}

const uint32_t SiStripRegionCabling::region(const Position position) const {
  PositionIndex index = positionIndex(position); 
  return region(index);
}

SiStripRegionCabling::PositionIndex SiStripRegionCabling::increment(const PositionIndex index, int deta, int dphi) const {
  
  int eta = static_cast<int>(index.first)+deta;
  if (eta > etadivisions_-1) eta = etadivisions_-1;
  else if (eta < 0) eta = 0;
 
  int phi = static_cast<int>(index.second)+dphi;
  while (phi<0) phi+=phidivisions_;
  while (phi>phidivisions_-1) phi-=phidivisions_;

  return PositionIndex(static_cast<uint32_t>(eta),static_cast<uint32_t>(phi));
}

const SiStripRegionCabling::SubDet SiStripRegionCabling::subdetFromDetId(const uint32_t detid) {
  DetId detectorId=DetId(detid);
  if (detectorId.subdetId() == StripSubdetector::TIB) return TIB;
  else if (detectorId.subdetId() == StripSubdetector::TID) return TID;
  else if (detectorId.subdetId() == StripSubdetector::TOB) return TOB;
  else if (detectorId.subdetId() == StripSubdetector::TEC) return TEC;
  else return ALLSUBDETS;
}

// -----------------------------------------------------------------------------
//
void SiStripRegionCabling::print( std::stringstream& ss ) const {
  uint32_t valid = 0;
  uint32_t total = 0;
  ss << "[SiStripRegionCabling::" << __func__ << "] Printing REGION cabling:" << std::endl;
  ss << "Printing cabling for " << regioncabling_.size() << " regions" << std::endl;
  Cabling::const_iterator id = regioncabling_.begin();
  Cabling::const_iterator jd = regioncabling_.end();
  for ( ; id != jd; ++id ) {
    ss << "Printing cabling for " << id->size()
       << " regions for partition " << static_cast<int32_t>( id - regioncabling_.begin() )
       << std::endl;
    RegionCabling::const_iterator ir = id->begin();
    RegionCabling::const_iterator jr = id->end();
    for ( ; ir != jr; ++ir ) {
      ss << "Printing cabling for " << ir->size()
	 << " wedges for region " << static_cast<int32_t>( ir - id->begin() )
	 << std::endl;
      WedgeCabling::const_iterator iw = ir->begin();
      WedgeCabling::const_iterator jw = ir->end();
      for ( ; iw != jw; ++iw ) {
	ss << "Printing cabling for " << iw->size()
	   << " elements for wedge " << static_cast<int32_t>( iw - ir->begin() )
	   << std::endl;
	ElementCabling::const_iterator ie = iw->begin();
	ElementCabling::const_iterator je = iw->end();
	for ( ; ie != je; ++ie ) {
	  ss << "Printing cabling for " << ie->second.size()
	     << " connections for element (DetId) " << ie->first
	     << std::endl;
	  std::vector<FedChannelConnection>::const_iterator ic = ie->second.begin();
	  std::vector<FedChannelConnection>::const_iterator jc = ie->second.end();
	  for ( ; ic != jc; ++ic ) {
	    if ( ic->isConnected() ) { valid++; }
	    total++;
	    ic->print(ss);
	    ss << std::endl;
	  }
	}
      }
    }
  }
  ss << "Number of connected:   " << valid << std::endl
     << "Number of connections: " << total << std::endl;
}


