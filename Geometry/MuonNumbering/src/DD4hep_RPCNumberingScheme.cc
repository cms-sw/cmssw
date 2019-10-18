
#include "Geometry/MuonNumbering/interface/DD4hep_RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include <FWCore/Utilities/interface/Exception.h>
#include <cassert>

using namespace cms;

RPCNumberingScheme::RPCNumberingScheme(const MuonConstants& muonConstants) { initMe(muonConstants); }

void RPCNumberingScheme::initMe(const MuonConstants& muonConstants) {
  int levelPart = get("level", muonConstants);

  assert(levelPart != 0);
  theRegionLevel = get("mr_region", muonConstants) / levelPart;
  theBWheelLevel = get("mr_bwheel", muonConstants) / levelPart;
  theBStationLevel = get("mr_bstation", muonConstants) / levelPart;
  theBPlaneLevel = get("mr_bplane", muonConstants) / levelPart;
  theBChamberLevel = get("mr_bchamber", muonConstants) / levelPart;
  theEPlaneLevel = get("mr_eplane", muonConstants) / levelPart;
  theESectorLevel = get("mr_esector", muonConstants) / levelPart;
  theERollLevel = get("mr_eroll", muonConstants) / levelPart;
}

void RPCNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {
  const int barrel = num.getSuperNo(theRegionLevel);

  bool barrel_muon = (barrel == 1);
  int maxLevel;
  if (barrel_muon) {
    maxLevel = theBChamberLevel;
  } else {
    maxLevel = theERollLevel;
  }
  if (num.getLevels() != maxLevel) {
    throw cms::Exception("DD4hep_RPCNumberingScheme", "num.getLevels() != maxLevel");
  }

  int plane_id = 0;
  int sector_id = 0;
  int copy_id = 0;
  int roll_id = 0;
  int eta_id = 0;
  int rr12_id = 0;
  bool forward = false;
  int sector_copy = 0;

  for (int level = 1; level <= maxLevel; level++) {
    if (level == theRegionLevel) {
      if (barrel_muon) {
        roll_id = 0;
      } else {
        copy_id = 1;
      }
    }
    if (barrel_muon) {
      if (level == theBWheelLevel) {
        const int copyno = num.getBaseNo(level);
        eta_id = 4 + copyno;
      } else if (level == theBStationLevel) {
        const int copyno = num.getBaseNo(level);
        sector_id = copyno + 1;
        if (sector_id == 13) {
          sector_id = 4;
          sector_copy = 1;
        } else if (sector_id == 14) {
          sector_id = 10;
          sector_copy = 1;
        }
        sector_id *= 3;
      } else if (level == theBPlaneLevel) {
        const int plane_tag = num.getSuperNo(level);
        if (plane_tag == 1) {
          plane_id = 1;
        } else if (plane_tag == 2) {
          plane_id = 5;
        } else if (plane_tag == 3) {
          plane_id = 2;
        } else if (plane_tag == 4) {
          plane_id = 6;
        } else if (plane_tag == 5) {
          plane_id = 3;
        } else {
          plane_id = 4;
        }
      } else if (level == theBChamberLevel) {
        const int copyno = num.getBaseNo(level);
        if ((plane_id == 4) && (sector_id == 4 * 3)) {
          copy_id = sector_copy * 2 + copyno + 1;
        } else if ((plane_id == 4) && (sector_id == 10 * 3)) {
          copy_id = sector_copy + 1;
        } else {
          copy_id = copyno + 1;
        }
        const int rollno = num.getSuperNo(level);
        roll_id = rollno;
      }
    } else {
      if (level == theRegionLevel) {
        const int copyno = num.getBaseNo(level);
        forward = (copyno == 0);
      } else if (level == theEPlaneLevel) {
        const int plane_tag = num.getSuperNo(level);
        const int rr12_tag = num.getBaseNo(level);
        plane_id = plane_tag;
        rr12_id = rr12_tag;
      } else if (level == theESectorLevel) {
        const int copyno = num.getBaseNo(level);
        sector_id = copyno + 1;
        if (rr12_id == 1) {
          sector_id = sector_id * 2 - 1;
        } else if (rr12_id == 2) {
          sector_id = sector_id * 2;
        }
      } else if (level == theERollLevel) {
        const int copyno = num.getBaseNo(level);
        const int eta_tag = num.getSuperNo(level);
        if ((eta_tag == 1) || (eta_tag == 4) || (eta_tag == 7) || (eta_tag == 8)) {
          eta_id = 1;
        } else if ((eta_tag == 2) || (eta_tag == 5)) {
          eta_id = 2;
        } else if ((eta_tag == 3) || (eta_tag == 6)) {
          eta_id = 3;
        }
        if (forward)
          eta_id = 12 - eta_id;
        if ((eta_tag == 4) || (eta_tag == 7) || (eta_tag == 8)) {
          sector_id *= 2;
        }
        roll_id = copyno + 1;
      }
    }
  }
  int trIndex = (eta_id * 10000 + plane_id * 1000 + sector_id * 10 + copy_id) * 10 + roll_id;
  RPCDetId id;
  id.buildfromTrIndex(trIndex);
  setDetId(id.rawId());
}

const int RPCNumberingScheme::get(const char* key, const MuonConstants& muonConstants) const {
  int result(0);
  auto const& it = (muonConstants.find(key));
  if (it != end(muonConstants))
    result = it->second;
  return result;
}
