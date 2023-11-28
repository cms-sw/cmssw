#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRYDBEP_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRYDBEP_H 1

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"

#include "CondFormats/Alignment/interface/Alignments.h"

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

//Forward declaration

//
// class declaration
//

namespace calogeometryDBEPimpl {
  // For U::writeFlag() == true
  template <typename T, bool>
  struct GeometryTraits {
    using TokenType = edm::ESGetToken<CaloSubdetectorGeometry, typename T::AlignedRecord>;

    static TokenType makeToken(edm::ESConsumesCollectorT<typename T::AlignedRecord>& cc) {
      return cc.template consumes<CaloSubdetectorGeometry>(
          edm::ESInputTag{"", T::producerTag() + std::string("_master")});
    }
  };

  template <typename T>
  struct GeometryTraits<T, false> {
    using TokenType = edm::ESGetToken<PCaloGeometry, typename T::PGeometryRecord>;

    static TokenType makeToken(edm::ESConsumesCollectorT<typename T::AlignedRecord>& cc) {
      return cc.template consumesFrom<PCaloGeometry, typename T::PGeometryRecord>(edm::ESInputTag{});
    }
  };

  // For the case of non-existent AlignmentRecord
  //
  // SFINAE tricks to detect if T::AlignmentRecord exists. Note that
  // the declarations of the following are sufficient.
  template <typename T>
  std::false_type has_AlignmentRecord(...);
  template <typename T>
  std::true_type has_AlignmentRecord(typename T::AlignmentRecord*);

  template <typename T>
  struct HasAlignmentRecord {
    static constexpr bool value = std::is_same<decltype(has_AlignmentRecord<T>(nullptr)), std::true_type>::value;
  };

  // Then define tokens from alignment record
  template <typename T, bool = HasAlignmentRecord<T>::value>
  struct AlignmentTokens {
    edm::ESGetToken<Alignments, typename T::AlignmentRecord> alignments;
    edm::ESGetToken<Alignments, GlobalPositionRcd> globals;
  };
  template <typename T>
  struct AlignmentTokens<T, false> {};

  // Some partial specializations need additional tokens...
  template <typename T>
  struct AdditionalTokens {
    void makeTokens(edm::ESConsumesCollectorT<typename T::AlignedRecord>& cc) {}
  };
}  // namespace calogeometryDBEPimpl

template <class T, class U>
class CaloGeometryDBEP : public edm::ESProducer {
public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;
  typedef CaloCellGeometry::Tr3D Tr3D;

  using PtrType = std::unique_ptr<CaloSubdetectorGeometry>;
  typedef CaloSubdetectorGeometry::TrVec TrVec;
  typedef CaloSubdetectorGeometry::DimVec DimVec;
  typedef CaloSubdetectorGeometry::IVec IVec;

  CaloGeometryDBEP(const edm::ParameterSet& ps) : applyAlignment_(ps.getParameter<bool>("applyAlignment")) {
    auto cc = setWhatProduced(this,
                              &CaloGeometryDBEP<T, U>::produceAligned,
                              edm::es::Label(T::producerTag()));  //+std::string("TEST") ) ) ;

    if constexpr (calogeometryDBEPimpl::HasAlignmentRecord<T>::value) {
      if (applyAlignment_) {
        alignmentTokens_.alignments =
            cc.template consumesFrom<Alignments, typename T::AlignmentRecord>(edm::ESInputTag{});
        alignmentTokens_.globals = cc.template consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{});
      }
    }
    geometryToken_ = calogeometryDBEPimpl::GeometryTraits<T, U::writeFlag()>::makeToken(cc);

    additionalTokens_.makeTokens(cc);
  }

  ~CaloGeometryDBEP() override {}

  PtrType produceAligned(const typename T::AlignedRecord& iRecord) {
    const auto [alignPtr, globalPtr] = getAlignGlobal(iRecord);

    TrVec tvec;
    DimVec dvec;
    IVec ivec;
    std::vector<uint32_t> dins;

    if constexpr (U::writeFlag()) {
      const auto& pG = iRecord.get(geometryToken_);

      pG.getSummary(tvec, ivec, dvec, dins);

      U::write(tvec, dvec, ivec, T::dbString());
    } else {
      const auto& pG = iRecord.get(geometryToken_);

      tvec = pG.getTranslation();
      dvec = pG.getDimension();
      ivec = pG.getIndexes();
    }
    //*********************************************************************************************

    const unsigned int nTrParm(tvec.size() / T::k_NumberOfCellsForCorners);

    assert(dvec.size() == T::k_NumberOfShapes * T::k_NumberOfParametersPerShape);

    PtrType ptr = std::make_unique<T>();

    ptr->fillDefaultNamedParameters();

    ptr->allocateCorners(T::k_NumberOfCellsForCorners);

    ptr->allocatePar(dvec.size(), T::k_NumberOfParametersPerShape);

    for (unsigned int i(0); i != T::k_NumberOfCellsForCorners; ++i) {
      const unsigned int nPerShape(T::k_NumberOfParametersPerShape);
      DimVec dims;
      dims.reserve(nPerShape);

      const unsigned int indx(ivec.size() == 1 ? 0 : i);

      DimVec::const_iterator dsrc(dvec.begin() + ivec[indx] * nPerShape);

      for (unsigned int j(0); j != nPerShape; ++j) {
        dims.emplace_back(*dsrc);
        ++dsrc;
      }

      const CCGFloat* myParm(CaloCellGeometry::getParmPtr(dims, ptr->parMgr(), ptr->parVecVec()));

      const DetId id(T::DetIdType::detIdFromDenseIndex(i));

      const unsigned int iGlob(nullptr == globalPtr ? 0 : T::alignmentTransformIndexGlobal(id));

      assert(nullptr == globalPtr || iGlob < globalPtr->m_align.size());

      const AlignTransform* gt(nullptr == globalPtr ? nullptr : &globalPtr->m_align[iGlob]);

      assert(nullptr == gt || iGlob == T::alignmentTransformIndexGlobal(DetId(gt->rawId())));

      const unsigned int iLoc(nullptr == alignPtr ? 0 : T::alignmentTransformIndexLocal(id));

      assert(nullptr == alignPtr || iLoc < alignPtr->m_align.size());

      const AlignTransform* at(nullptr == alignPtr ? nullptr : &alignPtr->m_align[iLoc]);

      assert(nullptr == at || (T::alignmentTransformIndexLocal(DetId(at->rawId())) == iLoc));

      const CaloGenericDetId gId(id);

      Pt3D lRef;
      Pt3DVec lc(8, Pt3D(0, 0, 0));
      T::localCorners(lc, &dims.front(), i, lRef);

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
    }

    ptr->initializeParms();  // initializations; must happen after cells filled

    return ptr;
  }

private:
  std::tuple<const Alignments*, const Alignments*> getAlignGlobal(const typename T::AlignedRecord& iRecord) const {
    const Alignments* alignPtr(nullptr);
    const Alignments* globalPtr(nullptr);
    if constexpr (calogeometryDBEPimpl::HasAlignmentRecord<T>::value) {
      if (applyAlignment_)  // get ptr if necessary
      {
        const auto& alignments = iRecord.get(alignmentTokens_.alignments);
        // require expected size
        assert(alignments.m_align.size() == T::numberOfAlignments());
        alignPtr = &alignments;

        const auto& globals = iRecord.get(alignmentTokens_.globals);
        globalPtr = &globals;
      }
    }
    return std::make_tuple(alignPtr, globalPtr);
  }

  typename calogeometryDBEPimpl::AlignmentTokens<T> alignmentTokens_;
  typename calogeometryDBEPimpl::GeometryTraits<T, U::writeFlag()>::TokenType geometryToken_;
  typename calogeometryDBEPimpl::AdditionalTokens<T> additionalTokens_;
  bool applyAlignment_;
};

#endif
