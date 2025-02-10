#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CaloGeometryDBZdc.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

//#define EDM_ML_DEBUG

template class CaloGeometryDBEP<CastorGeometry, CaloGeometryDBReader>;

typedef CaloGeometryDBEP<CastorGeometry, CaloGeometryDBReader> CastorGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(CastorGeometryFromDBEP);

template <>
CaloGeometryDBEP<ZdcGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<ZdcGeometry, CaloGeometryDBReader>::produceAligned(const typename ZdcGeometry::AlignedRecord& iRecord) {
  const auto [alignPtr, globalPtr] = getAlignGlobal(iRecord);

  TrVec tvec;
  DimVec dvec;
  IVec ivec;
  std::vector<uint32_t> dins;

  const auto& pG = iRecord.get(geometryToken_);

  tvec = pG.getTranslation();
  dvec = pG.getDimension();
  ivec = pG.getIndexes();
  dins = pG.getDenseIndices();
  //*********************************************************************************************
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ZDCGeometry") << "ZDCGeometry sizes " << tvec.size() << ":" << dvec.size() << ":" << ivec.size()
                                  << ":" << dins.size();
#endif
  const auto& zdcTopology = iRecord.get(additionalTokens_.topology);

  assert(dvec.size() <= ZdcGeometry::k_NumberOfShapes * ZdcGeometry::k_NumberOfParametersPerShape);
  ZdcGeometry* zdcGeometry = new ZdcGeometry(&zdcTopology);
  PtrType ptr(zdcGeometry);

  if (!dins.empty()) {
    const unsigned int nTrParm(tvec.size() / zdcTopology.kSizeForDenseIndexing());

    assert(dvec.size() == ZdcGeometry::k_NumberOfShapes * ZdcGeometry::k_NumberOfParametersPerShape);

    ptr->fillDefaultNamedParameters();

    ptr->allocateCorners(zdcTopology.kSizeForDenseIndexing());
    ptr->allocatePar(zdcGeometry->numberOfShapes(), ZdcGeometry::k_NumberOfParametersPerShape);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZDCGeometry") << "ZDCGeometry:Allocate corners and Par";
#endif
    for (unsigned int i(0); i < dins.size(); ++i) {
      const unsigned int nPerShape(ZdcGeometry::k_NumberOfParametersPerShape);
      DimVec dims;
      dims.reserve(nPerShape);

      const unsigned int indx(ivec.size() == 1 ? 0 : i);

      DimVec::const_iterator dsrc(dvec.begin() + ivec[indx] * nPerShape);

      for (unsigned int j(0); j != nPerShape; ++j) {
        dims.emplace_back(*dsrc);
        ++dsrc;
      }

      const CCGFloat* myParm(CaloCellGeometry::getParmPtr(dims, ptr->parMgr(), ptr->parVecVec()));

      const DetId id(zdcTopology.denseId2detId(dins[i]));

      const unsigned int iGlob(nullptr == globalPtr ? 0 : ZdcGeometry::alignmentTransformIndexGlobal(id));

      assert(nullptr == globalPtr || iGlob < globalPtr->m_align.size());

      const AlignTransform* gt(nullptr == globalPtr ? nullptr : &globalPtr->m_align[iGlob]);

      assert(nullptr == gt || iGlob == ZdcGeometry::alignmentTransformIndexGlobal(DetId(gt->rawId())));

      const unsigned int iLoc(nullptr == alignPtr ? 0 : ZdcGeometry::alignmentTransformIndexLocal(id));

      assert(nullptr == alignPtr || iLoc < alignPtr->m_align.size());

      const AlignTransform* at(nullptr == alignPtr ? nullptr : &alignPtr->m_align[iLoc]);

      assert(nullptr == at || (ZdcGeometry::alignmentTransformIndexLocal(DetId(at->rawId())) == iLoc));

      const CaloGenericDetId gId(id);

      Pt3D lRef;
      Pt3DVec lc(8, Pt3D(0, 0, 0));
      zdcGeometry->localCorners(lc, &dims.front(), dins[i], lRef);

      const Pt3D lBck(0.25 * (lc[4] + lc[5] + lc[6] + lc[7]));  // ctr rear  face in local
      const Pt3D lCor(lc[0]);

      //----------------------------------- create transform from 6 numbers ---
      const unsigned int jj(i * nTrParm);
      Tr3D tr;
      const ROOT::Math::Translation3D tl(tvec[jj], tvec[jj + 1], tvec[jj + 2]);
      const ROOT::Math::EulerAngles ea(6 == nTrParm ? ROOT::Math::EulerAngles(tvec[jj + 3], tvec[jj + 4], tvec[jj + 5])
                                                    : ROOT::Math::EulerAngles());
      const ROOT::Math::Transform3D rt(ea, tl);
      double xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz;
      rt.GetComponents(xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz);
      tr = Tr3D(CLHEP::HepRep3x3(xx, xy, xz, yx, yy, yz, zx, zy, zz), CLHEP::Hep3Vector(dx, dy, dz));

      // now prepend alignment(s) for final transform
      const Tr3D atr(nullptr == at ? tr
                                   : (nullptr == gt ? at->transform() * tr : at->transform() * gt->transform() * tr));
      //--------------------------------- done making transform  ---------------

      const Pt3D gRef(atr * lRef);
      const GlobalPoint fCtr(gRef.x(), gRef.y(), gRef.z());
      const Pt3D gBck(atr * lBck);
      const GlobalPoint fBck(gBck.x(), gBck.y(), gBck.z());
      const Pt3D gCor(atr * lCor);
      const GlobalPoint fCor(gCor.x(), gCor.y(), gCor.z());

      assert(zdcTopology.detId2denseId(id) == dins[i]);

      ptr->newCell(fCtr, fBck, fCor, myParm, id);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ZDCGeometry") << "ZDCGeometry Insert cell " << i << ":" << HcalZDCDetId(id);
#endif
    }
  } else {
    const unsigned int nTrParm(tvec.size() / HcalZDCDetId::kSizeForDenseIndexingRun1);

    assert(dvec.size() == ZdcGeometry::k_NumberOfShapes * ZdcGeometry::k_NumberOfParametersPerShape);

    ptr->fillDefaultNamedParameters();
    ptr->allocateCorners(HcalZDCDetId::kSizeForDenseIndexingRun1);
    ptr->allocatePar(zdcGeometry->numberOfShapes(), ZdcGeometry::k_NumberOfParametersPerShape);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZDCGeometry") << "ZDCGeometry:Allocate corners and Par";
#endif
    for (unsigned int i(0); i != HcalZDCDetId::kSizeForDenseIndexingRun1; ++i) {
      const unsigned int nPerShape(ZdcGeometry::k_NumberOfParametersPerShape);
      DimVec dims;
      dims.reserve(nPerShape);

      const unsigned int indx(ivec.size() == 1 ? 0 : i);

      DimVec::const_iterator dsrc(dvec.begin() + ivec[indx] * nPerShape);

      for (unsigned int j(0); j != nPerShape; ++j) {
        dims.emplace_back(*dsrc);
        ++dsrc;
      }

      const CCGFloat* myParm(CaloCellGeometry::getParmPtr(dims, ptr->parMgr(), ptr->parVecVec()));

      const DetId id(zdcTopology.denseId2detId(i));

      const unsigned int iGlob(nullptr == globalPtr ? 0 : ZdcGeometry::alignmentTransformIndexGlobal(id));

      assert(nullptr == globalPtr || iGlob < globalPtr->m_align.size());

      const AlignTransform* gt(nullptr == globalPtr ? nullptr : &globalPtr->m_align[iGlob]);

      assert(nullptr == gt || iGlob == ZdcGeometry::alignmentTransformIndexGlobal(DetId(gt->rawId())));

      const unsigned int iLoc(nullptr == alignPtr ? 0 : ZdcGeometry::alignmentTransformIndexLocal(id));

      assert(nullptr == alignPtr || iLoc < alignPtr->m_align.size());

      const AlignTransform* at(nullptr == alignPtr ? nullptr : &alignPtr->m_align[iLoc]);

      assert(nullptr == at || (ZdcGeometry::alignmentTransformIndexLocal(DetId(at->rawId())) == iLoc));

      const CaloGenericDetId gId(id);

      Pt3D lRef;
      Pt3DVec lc(8, Pt3D(0, 0, 0));
      zdcGeometry->localCorners(lc, &dims.front(), i, lRef);
      const Pt3D lBck(0.25 * (lc[4] + lc[5] + lc[6] + lc[7]));  // ctr rear  face in local
      const Pt3D lCor(lc[0]);

      //----------------------------------- create transform from 6 numbers ---
      const unsigned int jj(i * nTrParm);
      Tr3D tr;
      const ROOT::Math::Translation3D tl(tvec[jj], tvec[jj + 1], tvec[jj + 2]);
      const ROOT::Math::EulerAngles ea(6 == nTrParm ? ROOT::Math::EulerAngles(tvec[jj + 3], tvec[jj + 4], tvec[jj + 5])
                                                    : ROOT::Math::EulerAngles());
      const ROOT::Math::Transform3D rt(ea, tl);
      double xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz;
      rt.GetComponents(xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz);
      tr = Tr3D(CLHEP::HepRep3x3(xx, xy, xz, yx, yy, yz, zx, zy, zz), CLHEP::Hep3Vector(dx, dy, dz));

      // now prepend alignment(s) for final transform
      const Tr3D atr(nullptr == at ? tr
                                   : (nullptr == gt ? at->transform() * tr : at->transform() * gt->transform() * tr));
      //--------------------------------- done making transform  ---------------

      const Pt3D gRef(atr * lRef);
      const GlobalPoint fCtr(gRef.x(), gRef.y(), gRef.z());
      const Pt3D gBck(atr * lBck);
      const GlobalPoint fBck(gBck.x(), gBck.y(), gBck.z());
      const Pt3D gCor(atr * lCor);
      const GlobalPoint fCor(gCor.x(), gCor.y(), gCor.z());

      ptr->newCell(fCtr, fBck, fCor, myParm, id);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ZDCGeometry") << "ZDCGeometry Insert cell " << i << ":" << HcalZDCDetId(id);
#endif
    }
  }
  ptr->initializeParms();  // initializations; must happen after cells filled

  return ptr;
}

template class CaloGeometryDBEP<ZdcGeometry, CaloGeometryDBReader>;

typedef CaloGeometryDBEP<ZdcGeometry, CaloGeometryDBReader> ZdcGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(ZdcGeometryFromDBEP);
