#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H

#include <memory>
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "CondFormats/Alignment/interface/Alignments.h"

template <class T, class D>
class CaloGeometryEP : public edm::ESProducer {
public:
  using LoaderType = CaloGeometryLoader<T>;
  using PtrType = typename LoaderType::PtrType;

  CaloGeometryEP(const edm::ParameterSet& ps) : applyAlignment_(ps.getParameter<bool>("applyAlignment")) {
    auto cc = setWhatProduced(this, &CaloGeometryEP<T, D>::produceAligned, edm::es::Label(T::producerTag()));

    if (applyAlignment_) {
      alignmentsToken_ = cc.template consumesFrom<Alignments, typename T::AlignmentRecord>(edm::ESInputTag{});
      globalsToken_ = cc.template consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{});
    }
    cpvToken_ = cc.template consumesFrom<D, IdealGeometryRecord>(edm::ESInputTag{});
  }

  ~CaloGeometryEP() override {}
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
    edm::ESTransientHandle<D> cpv = iRecord.getTransientHandle(cpvToken_);

    LoaderType loader;
    return loader.load(cpv.product(), alignPtr, globalPtr);
  }

private:
  edm::ESGetToken<Alignments, typename T::AlignmentRecord> alignmentsToken_;
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalsToken_;
  edm::ESGetToken<D, IdealGeometryRecord> cpvToken_;
  bool applyAlignment_;
};

#endif
