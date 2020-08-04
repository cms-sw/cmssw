#ifndef GEOMETRY_VERYFORWARDGEOMETRYBUILDER_PPSGEOMETRYESPRODUCER_H
#define GEOMETRY_VERYFORWARDGEOMETRYBUILDER_PPSGEOMETRYESPRODUCER_H


/****************************************************************************
*
* Author:
*
*  Wagner Carvalho (wcarvalh@cern.ch)
*
*  Based on CTPPSGeometryESModule.cc by:
*
*  Jan Kaspar (jan.kaspar@gmail.com)
*  Dominik Mierzejewski <dmierzej@cern.ch>
*
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"


/**
 * \brief Builds ideal, real and misaligned geometries.
 *
 * First, it creates a tree of DetGeomDesc from DDCompView. For real and misaligned geometries,
 * it applies alignment corrections (RPAlignmentCorrections) found in corresponding ...GeometryRecord.
 *
 * Second, it creates CTPPSGeometry from DetGeoDesc tree.
 **/

  
class PPSGeometryESProducer : public edm::ESProducer {
 public:
  PPSGeometryESProducer(const edm::ParameterSet&);
  ~PPSGeometryESProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<DetGeomDesc> produceIdealGD(const IdealGeometryRecord&);
  std::vector<int> fillCopyNos(TGeoIterator& it);

  template <typename ALIGNMENT_REC>
  struct GDTokens {
    explicit GDTokens(edm::ESConsumesCollector&& iCC)
        : idealGDToken_{iCC.consumesFrom<DetGeomDesc, IdealGeometryRecord>(edm::ESInputTag())},
          alignmentToken_{iCC.consumesFrom<CTPPSRPAlignmentCorrectionsData, ALIGNMENT_REC>(edm::ESInputTag())} {}
    const edm::ESGetToken<DetGeomDesc, IdealGeometryRecord> idealGDToken_;
    const edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, ALIGNMENT_REC> alignmentToken_;
  };

  std::unique_ptr<DetGeomDesc> produceRealGD(const VeryForwardRealGeometryRecord&);
  std::unique_ptr<CTPPSGeometry> produceRealTG(const VeryForwardRealGeometryRecord&);

  std::unique_ptr<DetGeomDesc> produceMisalignedGD(const VeryForwardMisalignedGeometryRecord&);
  std::unique_ptr<CTPPSGeometry> produceMisalignedTG(const VeryForwardMisalignedGeometryRecord&);

  template <typename REC>
  std::unique_ptr<DetGeomDesc> produceGD(IdealGeometryRecord const&,
                                         const std::optional<REC>&,
                                         GDTokens<REC> const&,
                                         const char* name);

  static void applyAlignments(const DetGeomDesc&, const CTPPSRPAlignmentCorrectionsData*, DetGeomDesc*&);

  const unsigned int verbosity_;
  const edm::ESGetToken<cms::DDDetector, IdealGeometryRecord> detectorToken_;

  const GDTokens<RPRealAlignmentRecord> gdRealTokens_;
  const GDTokens<RPMisalignedAlignmentRecord> gdMisTokens_;

  const edm::ESGetToken<DetGeomDesc, VeryForwardRealGeometryRecord> dgdRealToken_;
  const edm::ESGetToken<DetGeomDesc, VeryForwardMisalignedGeometryRecord> dgdMisToken_;
};

#endif
