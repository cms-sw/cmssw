#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

size_t MTDTopology::hshiftETL(const uint32_t detid, const int horizontalShift) const {
  ETLDetId start_mod(detid);

  if (horizontalShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null horizotalShift in ETL";
    return failIndex_;
  }
  int hsh = horizontalShift > 0 ? 1 : -1;

  int module = start_mod.module();
  uint32_t modtyp = start_mod.modType();
  uint32_t discside = start_mod.discSide();

  // ilayout number coincides at present with disc face, use this

  size_t iHome = (modtyp == etlVals_[discside].idDetType1_) ? 0 : 1;
  size_t iLeft = (etlVals_[discside].idDetType1_ == 1) ? 0 : 1;

  // for left type modules the position according to the default order is module - 1, for the rigth type modules the total number of left modules must be added

  size_t nmodOffset = (modtyp == 1) ? 0 : etlVals_[discside].start_copy_[iLeft].back() - 1;

  for (size_t iloop = 0; iloop < etlVals_[discside].start_copy_[iHome].size() - 1; iloop++) {
    if (module >= etlVals_[discside].start_copy_[iHome][iloop] &&
        module < etlVals_[discside].start_copy_[iHome][iloop + 1]) {
      if (module + hsh >= etlVals_[discside].start_copy_[iHome][iloop] &&
          module + hsh < etlVals_[discside].start_copy_[iHome][iloop + 1]) {
        return module + hsh - 1 + nmodOffset;
      }
      break;
    }
  }

  return failIndex_;
}

size_t MTDTopology::vshiftETL(const uint32_t detid, const int verticalShift, size_t& closest) const {
  closest = failIndex_;

  ETLDetId start_mod(detid);

  if (verticalShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null verticalShift in ETL";
    return failIndex_;
  }
  int vsh = verticalShift > 0 ? 1 : -1;

  int module = start_mod.module();
  uint32_t modtyp = start_mod.modType();
  uint32_t discside = start_mod.discSide();

  // ilayout number coincides at present with disc face, use this

  size_t iHome = (modtyp == etlVals_[discside].idDetType1_) ? 0 : 1;
  size_t iOther = (iHome == 0) ? 1 : 0;
  size_t iLeft = (etlVals_[discside].idDetType1_ == 1) ? 0 : 1;

  // for right type modules the offset of the total number of left modules needs to be added,
  // what matters here is the other type, i.e. if the starting module is left the vertical shift moves towards a right type, and viceversa

  size_t nmodOffset = (modtyp == 1) ? etlVals_[discside].start_copy_[iLeft].back() - 1 : 0;

  size_t iBin(etlVals_[discside].start_copy_[iHome].size());  // never allowed
  for (size_t iloop = 0; iloop < etlVals_[discside].start_copy_[iHome].size() - 1; iloop++) {
    if (module >= etlVals_[discside].start_copy_[iHome][iloop] &&
        module < etlVals_[discside].start_copy_[iHome][iloop + 1]) {
      iBin = iloop;
      break;
    }
  }

  if (iBin == etlVals_[discside].start_copy_[iHome].size()) {
    edm::LogWarning("MTDTopology") << "Module number not compatible with layout, abort";
    return failIndex_;
  }

  // define the interval of interest for the other type according to the vertical shift sign

  int iBinOther(iBin);
  if (iHome == 0 && vsh < 0) {
    iBinOther = iBin - 1;
  }
  if (iHome == 1 && vsh > 0) {
    iBinOther = iBin + 1;
  }
  if (iBinOther < 0 || iBinOther >= static_cast<int>(etlVals_[discside].start_copy_[iOther].size()) - 1) {
    return failIndex_;
  }

  // determine the position of the other type corresponding to the same column of the home type

  int vpos = etlVals_[discside].offset_[iHome][iBin] + module - etlVals_[discside].start_copy_[iHome][iBin] + 1;
  if (vpos <= etlVals_[discside].offset_[iOther][iBinOther]) {
    closest = etlVals_[discside].start_copy_[iOther][iBinOther];
  } else if (vpos > etlVals_[discside].offset_[iOther][iBinOther] +
                        etlVals_[discside].start_copy_[iOther][iBinOther + 1] -
                        etlVals_[discside].start_copy_[iOther][iBinOther] ||
             (vpos == etlVals_[discside].offset_[iOther][iBinOther] +
                          etlVals_[discside].start_copy_[iOther][iBinOther + 1] -
                          etlVals_[discside].start_copy_[iOther][iBinOther] &&
              iBinOther + 1 == static_cast<int>(etlVals_[discside].start_copy_[iOther].size()))) {
    closest = etlVals_[discside].start_copy_[iOther][iBinOther + 1] - 1;
  }
  if (closest < failIndex_) {
    closest = closest + nmodOffset - 1;
    return failIndex_;
  } else {
    // number of module shifted by 1 wrt the position in the array (i.e. module 1 has index 0)
    return etlVals_[discside].start_copy_[iOther][iBinOther] + vpos - 1 -
           etlVals_[discside].offset_[iOther][iBinOther] + nmodOffset - 1;
  }
}
