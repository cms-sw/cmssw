#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef CaloCellGeometry::Pt3D Pt3D;
typedef CaloCellGeometry::Pt3DVec Pt3DVec;
typedef HepGeom::Plane3D<CCGFloat> Pl3D;

//#define EDM_ML_DEBUG

EcalPreshowerGeometry::EcalPreshowerGeometry()
    : m_xWidWaf(6.3),
      m_xInterLadGap(0.05),  // additional gap between wafers in adj ladders
      m_xIntraLadGap(0.04),  // gap between wafers in same ladder
      m_yWidAct(6.1),
      m_yCtrOff(0.05),  // gap at center
      m_cellVec(k_NumberOfCellsForCorners) {
  m_zplane[0] = 0.;
  m_zplane[1] = 0.;
  m_zplane[2] = 0.;
  m_zplane[3] = 0.;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalPreshowerGeometry::Creating an instance";
#endif
}

EcalPreshowerGeometry::~EcalPreshowerGeometry() {}

unsigned int EcalPreshowerGeometry::alignmentTransformIndexLocal(const DetId& id) {
  const CaloGenericDetId gid(id);

  assert(gid.isES());

  // plane 2 is split into 2 dees along x=0 for both signs of z

  // plane 1 at zsign=-1 is split into 2 dees between six=19 and six=20 for siy<=20,
  //                                             and six=21 and 22 for siy>=21

  // plane 1 at zsign=+1 is split into 2 dees between six=20 and six=21 for siy<=20,
  //                                             and six=19 and 20 for siy>=21

  // Desired numbering
  //                LEFT    RIGHT (as one faces the Dee from the IP)
  //  ES-  pl=2     0       1
  //       pl=1     2       3    the reversal of pl=2 and pl=1 is intentional here (CM Kuo)
  //  ES+  pl=1     4       5
  //       pl=2     6       7

  const ESDetId esid(id);
  const int jx(esid.six() - 1);
  const int jy(esid.siy() - 1);
  const int jz(esid.zside() + 1);
  const int pl(esid.plane() - 1);
  const bool second(1 == pl);
  const bool top(19 < jy);
  const bool negz(0 == jz);
  const int lrl(19 > jx ? 0 : 1);
  const int lrr(21 > jx ? 0 : 1);

  return (second ? jx / 20 + 3 * jz :        // 2nd plane split along middle
              (negz && !top ? lrl + 2 :      // 1st plane at neg z and bottom half split at six=19&20
                   (negz && top ? lrr + 2 :  // 1st plane at neg z and top half split at six=21&22
                        (!negz && !top ? lrr + 4 : lrl + 4))));  // opposite at positive z
}

DetId EcalPreshowerGeometry::detIdFromLocalAlignmentIndex(unsigned int iLoc) {
  return ESDetId(1, 10 + 20 * (iLoc % 2), 10, 2 > iLoc || 5 < iLoc ? 2 : 1, 2 * (iLoc / 4) - 1);
}

unsigned int EcalPreshowerGeometry::alignmentTransformIndexGlobal(const DetId& /*id*/) {
  return (unsigned int)DetId::Ecal - 1;
}

void EcalPreshowerGeometry::initializeParms() {
  unsigned int n1minus(0);
  unsigned int n2minus(0);
  unsigned int n1plus(0);
  unsigned int n2plus(0);
  CCGFloat z1minus(0);
  CCGFloat z2minus(0);
  CCGFloat z1plus(0);
  CCGFloat z2plus(0);
  const std::vector<DetId>& esDetIds(getValidDetIds());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalPreshowerGeometry:: Get " << esDetIds.size() << " valid DetIds";
#endif

  for (unsigned int i(0); i != esDetIds.size(); ++i) {
    const ESDetId esid(esDetIds[i]);
    auto cell = getGeometry(esid);
    if (nullptr != cell) {
      const CCGFloat zz(cell->getPosition().z());
      if (1 == esid.plane()) {
        if (0 > esid.zside()) {
          z1minus += zz;
          ++n1minus;
        } else {
          z1plus += zz;
          ++n1plus;
        }
      }
      if (2 == esid.plane()) {
        if (0 > esid.zside()) {
          z2minus += zz;
          ++n2minus;
        } else {
          z2plus += zz;
          ++n2plus;
        }
      }
    }
  }
  assert(0 != n1minus && 0 != n2minus && 0 != n1plus && 0 != n2plus);
  z1minus /= (1. * n1minus);
  z2minus /= (1. * n2minus);
  z1plus /= (1. * n1plus);
  z2plus /= (1. * n2plus);
  assert(0 != z1minus && 0 != z2minus && 0 != z1plus && 0 != z2plus);
  setzPlanes(z1minus, z2minus, z1plus, z2plus);
}

void EcalPreshowerGeometry::setzPlanes(CCGFloat z1minus, CCGFloat z2minus, CCGFloat z1plus, CCGFloat z2plus) {
  assert(0 > z1minus && 0 > z2minus && 0 < z1plus && 0 < z2plus);

  m_zplane[0] = z1minus;
  m_zplane[1] = z2minus;
  m_zplane[2] = z1plus;
  m_zplane[3] = z2plus;
}

// Get closest cell, etc...
DetId EcalPreshowerGeometry::getClosestCell(const GlobalPoint& point) const { return getClosestCellInPlane(point, 2); }

DetId EcalPreshowerGeometry::getClosestCellInPlane(const GlobalPoint& point, int plane) const {
  const CCGFloat x(point.x());
  const CCGFloat y(point.y());
  const CCGFloat z(point.z());

  if (0 == z || 1 > plane || 2 < plane)
    return DetId(0);

  const unsigned int iz((0 > z ? 0 : 2) + plane - 1);

  const CCGFloat ze(m_zplane[iz]);
  const CCGFloat xe(x * ze / z);
  const CCGFloat ye(y * ze / z);

  const CCGFloat x0(1 == plane ? xe : ye);
  const CCGFloat y0(1 == plane ? ye : xe);

  static const CCGFloat xWid(m_xWidWaf + m_xIntraLadGap + m_xInterLadGap);

  const int row(1 + int(y0 + 20. * m_yWidAct - m_yCtrOff) / m_yWidAct);
  const int col(1 + int((x0 + 20. * xWid) / xWid));

  CCGFloat closest(1e9);

  DetId detId(0);

  const int jz(0 > ze ? -1 : 1);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "** p=" << point << ", (" << xe << ", " << ye << ", " << ze << "), row=" << row
                               << ", col=" << col;
#endif
  for (int ix(-1); ix != 2; ++ix)  // search within +-1 in row and col
  {
    for (int iy(-1); iy != 2; ++iy) {
      for (int jstrip(ESDetId::ISTRIP_MIN); jstrip <= ESDetId::ISTRIP_MAX; ++jstrip) {
        const int jx(1 == plane ? col + ix : row + iy);
        const int jy(1 == plane ? row + iy : col + ix);
        if (ESDetId::validDetId(jstrip, jx, jy, plane, jz)) {
          const ESDetId esId(jstrip, jx, jy, plane, jz);
          auto cell = getGeometry(esId);
          if (nullptr != cell) {
            const GlobalPoint& p(cell->getPosition());
            const CCGFloat dist2((p.x() - xe) * (p.x() - xe) + (p.y() - ye) * (p.y() - ye));
            if (dist2 < closest && present(esId)) {
              closest = dist2;
              detId = esId;
            }
          }
        }
      }
    }
  }
  return detId;
}

void EcalPreshowerGeometry::localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int /*i*/, Pt3D& ref) {
  PreshowerStrip::localCorners(lc, pv, ref);
}

void EcalPreshowerGeometry::newCell(const GlobalPoint& f1,
                                    const GlobalPoint& /*f2*/,
                                    const GlobalPoint& /*f3*/,
                                    const CCGFloat* parm,
                                    const DetId& detId) {
  const unsigned int cellIndex(ESDetId(detId).denseIndex());
  m_cellVec[cellIndex] = PreshowerStrip(f1, cornersMgr(), parm);
  addValidID(detId);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalPreshowerGeometry::Add a new DetId " << ESDetId(detId);
#endif
}

const CaloCellGeometry* EcalPreshowerGeometry::getGeometryRawPtr(uint32_t index) const {
  // Modify the RawPtr class
  const CaloCellGeometry* cell(&m_cellVec[index]);
  return (m_cellVec.size() < index || nullptr == cell->param() ? nullptr : cell);
}
