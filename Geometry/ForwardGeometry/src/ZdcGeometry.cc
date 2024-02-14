#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/ZdcHardcodeGeometryData.h"
#include <algorithm>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

#include <algorithm>

typedef CaloCellGeometry::Pt3D Pt3D;
typedef CaloCellGeometry::Pt3DVec Pt3DVec;
typedef CaloCellGeometry::Tr3D Tr3D;

typedef CaloSubdetectorGeometry::CCGFloat CCGFloat;

//#define EDM_ML_DEBUG

ZdcGeometry::ZdcGeometry()
    : theTopology(new ZdcTopology),
      lastReqDet_(DetId::Detector(0)),
      lastReqSubdet_(0),
      m_ownsTopology(true),
      m_cellVec(k_NumberOfCellsForCorners) {}

ZdcGeometry::ZdcGeometry(const ZdcTopology* topology)
    : theTopology(topology),
      lastReqDet_(DetId::Detector(0)),
      lastReqSubdet_(0),
      m_ownsTopology(false),
      m_cellVec(k_NumberOfCellsForCorners) {}

ZdcGeometry::~ZdcGeometry() {
  if (m_ownsTopology)
    delete theTopology;
}
/*
DetId ZdcGeometry::getClosestCell(const GlobalPoint& r) const
{
   DetId returnId ( 0 ) ;
   const std::vector<DetId>& detIds ( getValidDetIds() ) ;
   for( std::vector<DetId>::const_iterator it ( detIds.begin() ) ;
	it != detIds.end(); ++it )
   {
      auto cell = ( getGeometry( *it ) ) ;
      if( 0 != cell &&
	  cell->inside( r ) )
      {
	 returnId = *it ;
	 break ;
      }
   }
   return returnId ;
}
*/
unsigned int ZdcGeometry::alignmentTransformIndexLocal(const DetId& id) {
  const CaloGenericDetId gid(id);
  assert(gid.isZDC());

  return (0 > HcalZDCDetId(id).zside() ? 0 : 1);
}

unsigned int ZdcGeometry::alignmentTransformIndexGlobal(const DetId& /*id*/) { return (unsigned int)DetId::Calo - 1; }

void ZdcGeometry::localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int /*i*/, Pt3D& ref) {
  IdealZDCTrapezoid::localCorners(lc, pv, ref);
}

void ZdcGeometry::newCell(const GlobalPoint& f1,
                          const GlobalPoint& /*f2*/,
                          const GlobalPoint& /*f3*/,
                          const CCGFloat* parm,
                          const DetId& detId) {
  const CaloGenericDetId cgid(detId);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ZDCGeom") << "ZDCGeometry " << HcalZDCDetId(detId) << " Generic ID " << std::hex << cgid.rawId()
                              << std::dec << " ZDC? " << cgid.isZDC();
#endif
  assert(cgid.isZDC());

  const unsigned int di(cgid.denseIndex());

  m_cellVec[di] = IdealZDCTrapezoid(f1, cornersMgr(), parm);
  addValidID(detId);
}

const CaloCellGeometry* ZdcGeometry::getGeometryRawPtr(uint32_t index) const {
  // Modify the RawPtr class
  if (m_cellVec.size() < index)
    return nullptr;
  const CaloCellGeometry* cell(&m_cellVec[index]);
  return (((cell == nullptr) || (nullptr == cell->param())) ? nullptr : cell);
}

void ZdcGeometry::getSummary(CaloSubdetectorGeometry::TrVec& tVec,
                             CaloSubdetectorGeometry::IVec& iVec,
                             CaloSubdetectorGeometry::DimVec& dVec,
                             CaloSubdetectorGeometry::IVec& /*dins*/) const {
  tVec.reserve(m_validIds.size() * numberOfTransformParms());
  iVec.reserve(numberOfShapes() == 1 ? 1 : m_validIds.size());
  dVec.reserve(numberOfShapes() * numberOfParametersPerShape());

  for (const auto& pv : parVecVec()) {
    for (float iv : pv) {
      dVec.emplace_back(iv);
    }
  }

  for (uint32_t i(0); i != m_validIds.size(); ++i) {
    Tr3D tr;
    std::shared_ptr<const CaloCellGeometry> ptr(cellGeomPtr(i));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZDCGeom") << "ZDCGeometry:Summary " << i << ":" << HcalZDCDetId::kSizeForDenseIndexingRun1
                                << " Pointer " << ptr << ":" << (nullptr != ptr);
#endif
    if (i < static_cast<int32_t>(HcalZDCDetId::kSizeForDenseIndexingRun1))
      assert(nullptr != ptr);
    if (ptr != nullptr) {
      ptr->getTransform(tr, (Pt3DVec*)nullptr);

      if (Tr3D() == tr) {  // for preshower there is no rotation
        const GlobalPoint& gp(ptr->getPosition());
        tr = HepGeom::Translate3D(gp.x(), gp.y(), gp.z());
      }

      const CLHEP::Hep3Vector tt(tr.getTranslation());
      tVec.emplace_back(tt.x());
      tVec.emplace_back(tt.y());
      tVec.emplace_back(tt.z());
      if (6 == numberOfTransformParms()) {
        const CLHEP::HepRotation rr(tr.getRotation());
        const ROOT::Math::Transform3D rtr(
            rr.xx(), rr.xy(), rr.xz(), tt.x(), rr.yx(), rr.yy(), rr.yz(), tt.y(), rr.zx(), rr.zy(), rr.zz(), tt.z());
        ROOT::Math::EulerAngles ea;
        rtr.GetRotation(ea);
        tVec.emplace_back(ea.Phi());
        tVec.emplace_back(ea.Theta());
        tVec.emplace_back(ea.Psi());
      }

      const CCGFloat* par(ptr->param());

      unsigned int ishape(9999);
      for (unsigned int ivv(0); ivv != parVecVec().size(); ++ivv) {
        bool ok(true);
        const CCGFloat* pv(&(*parVecVec()[ivv].begin()));
        for (unsigned int k(0); k != numberOfParametersPerShape(); ++k) {
          ok = ok && (fabs(par[k] - pv[k]) < 1.e-6);
        }
        if (ok) {
          ishape = ivv;
          break;
        }
      }
      assert(9999 != ishape);

      const unsigned int nn((numberOfShapes() == 1) ? (unsigned int)1 : m_validIds.size());
      if (iVec.size() < nn)
        iVec.emplace_back(ishape);
    }
  }
}
