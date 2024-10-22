///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

//#define EDM_ML_DEBUG

EcalEndcapNumberingScheme::EcalEndcapNumberingScheme() : EcalNumberingScheme() {
  edm::LogVerbatim("EcalGeom") << "Creating EcalEndcapNumberingScheme";
}

EcalEndcapNumberingScheme::~EcalEndcapNumberingScheme() {
  edm::LogVerbatim("EcalGeom") << "Deleting EcalEndcapNumberingScheme";
}
uint32_t EcalEndcapNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  const uint32_t nLevels(baseNumber.getLevels());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "ECalEndcapNumberingScheme geometry levels = " << nLevels;
  for (uint32_t k = 0; k < nLevels; ++k)
    edm::LogVerbatim("EcalGeom") << "[" << k << "] " << baseNumber.getLevelName(k) << ":"
                                 << baseNumber.getCopyNumber(k);
#endif
  if (7 > nLevels) {
    edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                << "Not enough levels found in EcalBaseNumber ( " << nLevels << ") Returning 0";
    return 0;
  }

  if (nLevels <= 10) {
    // Static geometry

    if (baseNumber.getLevels() < 1) {
      edm::LogWarning("EcalGeom") << "EalEndcaplNumberingScheme::getUnitID: No "
                                  << "level found in EcalBaseNumber Returning 0";
      return 0;
    }

    int PVid = baseNumber.getCopyNumber(0);
    int MVid = 1;
    if (baseNumber.getLevels() > 1)
      MVid = baseNumber.getCopyNumber(1);
    else
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID: Null"
                                  << " pointer to alveole ! Use default id=1";

    int zs = baseNumber.getCopyNumber("EREG");
    int zside = 2 * (1 - zs) + 1;
    int module_number = MVid;
    int crystal_number = PVid;

    uint32_t intindex = EEDetId(module_number, crystal_number, zside, EEDetId::SCCRYSTALMODE).rawId();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << "EcalEndcapNumberingScheme: zside = " << zs << ":" << zside
                                 << " super crystal = " << module_number << " crystal = " << crystal_number
                                 << " packed index = 0x" << std::hex << intindex << std::dec;
#endif
    return intindex;
  } else {
    // algorithmic geometry

    const int ic(baseNumber.getCopyNumber(0) % 100);  // crystal #, 0-44
    const int icx(ic / 10);
    const int icy(ic % 5);
    const int is(baseNumber.getCopyNumber(2) % 100);  // supercrystal #, 0-99
    const int isx(is / 10);
    const int isy(is % 10);

    const int iq(3 - 2 * baseNumber.getCopyNumber(3));  // quadrant #, -1, +1
    const int id(3 - 2 * baseNumber.getCopyNumber(5));  // dee      #, -1, +1

    const int iz(3 - 2 * baseNumber.getCopyNumber(7));  // z: -1, +1

    const int ix(50 + id * iz * (5 * isx + icx + 1) - (id * iz - 1) / 2);  // x: 1-100
    const int iy(50 + id * iq * (5 * isy + icy + 1) - (id * iq - 1) / 2);  // y: 1-100

    const uint32_t idet(DetId(DetId::Ecal, EEDetId::Subdet) | (((0 < iz ? 0x4000 : 0)) + (ix << 7) + iy));

    //*************************** ERROR CHECKING **********************************

    if (0 > icx || 4 < icx || 0 > icy || 4 < icy) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad crystal number = " << ic
                                  << ", Volume Name = " << baseNumber.getLevelName(0);
      return 0;
    }

    if (0 > isx || 9 < isx || 0 > isy || 9 < isy) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad supercrystal number = " << is
                                  << ", Volume Name = " << baseNumber.getLevelName(3);
      return 0;
    }

    if (1 != iq && -1 != iq) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad quadrant number = " << iq
                                  << ", Volume Name = " << baseNumber.getLevelName(4);
      return 0;
    }

    if (1 != id && -1 != id) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad dee number = " << id
                                  << ", Volume Name = " << baseNumber.getLevelName(6);
      return 0;
    }

    if (-1 != iz && 1 != iz) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad z-end number = " << iz
                                  << ", Volume Name = " << baseNumber.getLevelName(8);
      return 0;
    }

    if (!EEDetId::validDetId(ix, iy, iz)) {
      edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): "
                                  << "****************** Bad DetId numbers = " << ix << ", " << iy << ", " << iz;
      return 0;
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID(): " << std::dec << ix << ", " << iy << ", "
                                 << iq << ", " << id << ", " << iz << ", " << std::hex << idet << std::dec;
    edm::LogVerbatim("EcalGeom") << "ECalEndcapNumberingScheme::EEDetId: " << EEDetId(idet);
#endif
    return idet;
  }
}
