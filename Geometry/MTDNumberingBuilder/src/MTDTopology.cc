#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

MTDTopology::MTDTopology(const int& topologyMode, const ETLValues& etl)
    : mtdTopologyMode_(topologyMode), etlVals_(etl) {}

bool MTDTopology::orderETLSector(const GeomDet*& gd1, const GeomDet*& gd2) {
  ETLDetId det1(gd1->geographicalId().rawId());
  ETLDetId det2(gd2->geographicalId().rawId());

  if (det1.mtdRR() != det2.mtdRR()) {
    return det1.mtdRR() < det2.mtdRR();
  } else if (det1.modType() != det2.modType()) {
    return det1.modType() < det2.modType();
  } else {
    return det1.module() < det2.module();
  }
}

//size_t MTDTopology::hshiftETL(const uint32_t detid, const int horizonthalShift) const {

//size_t returnIndex(std::numeric_limits<unsigned int>::max()); // return out-of-range value for any failure

//ETLDetId start_mod(detid);

//if (horizontalShift == 0) {
//edm::LogWarning("MTDTopology") << "asking of a null horizotalShift in ETL";
//return returnIndex;
//}
//int hsh = horizonthalShift > 0 ? 1 : -1;

//uint32_t module = start_mod.module();
//uint32_t modtyp = start_mod.modType();
//uint32_t discside = start_mod.discSide();

//// ilayout number coincides at present with disc face, use this

//if ( etlVals_[discside].idDetType1_ == modtyp ) {
//for (size_t iloop = 0; iloop < start_copy_1_.size() - 1; iloop++ ) {
//if ( module >= start_copy_1_[iloop] && module < start_copy_1_[iloop+1] ) { ibin = iloop ; break; }
//}
//} else {
//for (size_t iloop = 0; iloop < start_copy_1_.size() - 1; iloop++ ) {
//if ( module >= start_copy_1_[iloop] && module < start_copy_1_[iloop+1] ) { ibin = iloop ; break; }
//}
//}
//}
