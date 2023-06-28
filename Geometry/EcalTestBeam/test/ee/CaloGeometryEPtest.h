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
#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondFormats/Alignment/interface/Alignments.h"

//Forward declaration

//
// class declaration
//

template <class T>
class CaloGeometryEPtest : public edm::ESProducer {
public:
  using LoaderType = CaloGeometryLoaderTest<T>;
  using PtrType = typename LoaderType::PtrType;

  CaloGeometryEPtest(const edm::ParameterSet& ps) : m_applyAlignment(ps.getParameter<bool>("applyAlignment")) {
    auto cc = setWhatProduced(this, &CaloGeometryEPtest<T>::produceAligned, edm::es::Label(T::producerTag()));
    if (m_applyAlignment) {
      m_alignmentsToken = cc.template consumesFrom<Alignments, typename T::AlignmentRecord>(edm::ESInputTag{});
      m_globalsToken = cc.template consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{});
    }
    m_geometryToken = cc.template consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{});
  }

  ~CaloGeometryEPtest() override {}
  PtrType produceAligned(const typename T::AlignedRecord& iRecord) {
    const Alignments* alignPtr(nullptr);
    const Alignments* globalPtr(nullptr);
    if (m_applyAlignment)  // get ptr if necessary
    {
      const auto& alignments = iRecord.get(m_alignmentsToken);

      // require expected size
      assert(alignments.m_align.size() == T::numberOfAlignments());
      alignPtr = &alignments;

      globalPtr = &(iRecord.get(m_globalsToken));
    }
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(m_geometryToken);

    LoaderType loader;
    return loader.load(cpv.product(), alignPtr, globalPtr);
  }

private:
  edm::ESGetToken<Alignments, typename T::AlignmentRecord> m_alignmentsToken;
  edm::ESGetToken<Alignments, GlobalPositionRcd> m_globalsToken;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> m_geometryToken;

  bool m_applyAlignment;
};

#endif
