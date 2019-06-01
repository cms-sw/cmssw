#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H 1

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondFormats/Alignment/interface/Alignments.h"

//Forward declaration

//
// class declaration
//

template <class T>
class CaloGeometryEP : public edm::ESProducer {
public:
  using LoaderType = CaloGeometryLoader<T>;
  using PtrType = typename LoaderType::PtrType;

  CaloGeometryEP<T>(const edm::ParameterSet& ps) : applyAlignment_(ps.getParameter<bool>("applyAlignment")) {
    auto cc = setWhatProduced(this,
                              &CaloGeometryEP<T>::produceAligned,
                              //                                  dependsOn( &CaloGeometryEP<T>::idealRecordCallBack ),
                              edm::es::Label(T::producerTag()));

    if (applyAlignment_) {
      alignmentsToken_ = cc.template consumesFrom<Alignments, typename T::AlignmentRecord>(edm::ESInputTag{});
      globalsToken_ = cc.template consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{});
    }
    cpvToken_ = cc.template consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{});
  }

  ~CaloGeometryEP<T>() override {}
  PtrType produceAligned(const typename T::AlignedRecord& iRecord) {
    const Alignments* alignPtr(nullptr);
    const Alignments* globalPtr(nullptr);
    if (applyAlignment_)  // get ptr if necessary
    {
      const auto& alignments = iRecord.get(alignmentsToken_);
      // require expected size
      assert(alignments.m_align.size() == T::numberOfAlignments());
      alignPtr = &alignments;

      const auto& globals = iRecord.get(globalsToken_);
      globalPtr = &globals;
    }
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvToken_);

    LoaderType loader;
    return loader.load(cpv.product(), alignPtr, globalPtr);
  }

private:
  edm::ESGetToken<Alignments, typename T::AlignmentRecord> alignmentsToken_;
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalsToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  bool applyAlignment_;
};

#endif
