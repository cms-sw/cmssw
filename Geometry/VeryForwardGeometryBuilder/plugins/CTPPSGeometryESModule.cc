/****************************************************************************
*  Based on CTPPSGeometryESModule.cc by:
*  Jan Kaspar (jan.kaspar@gmail.com)
*  Dominik Mierzejewski <dmierzej@cern.ch>
*
*  Rewritten + Moved out common functionailities to DetGeomDesc(Builder) by Gabrielle Hugo.
*  Migrated to DD4hep by Wagner Carvalho and Gabrielle Hugo.
*
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometryESCommon.h"

#include <regex>

/**
 * \brief Builds ideal, real and misaligned geometries.
 *
 * First, it creates a tree of DetGeomDesc from DDCompView. For real and misaligned geometries,
 * it applies alignment corrections (RPAlignmentCorrections) found in corresponding ...GeometryRecord.
 *
 * Second, it creates CTPPSGeometry from DetGeoDesc tree.
 **/

using RotationMatrix = ROOT::Math::Rotation3D;
using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

class CTPPSGeometryESModule : public edm::ESProducer {
public:
  CTPPSGeometryESModule(const edm::ParameterSet&);
  ~CTPPSGeometryESModule() override {}

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

  const unsigned int verbosity_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  const bool fromDD4hep_;

  const GDTokens<RPRealAlignmentRecord> gdRealTokens_;
  const GDTokens<RPMisalignedAlignmentRecord> gdMisTokens_;

  const edm::ESGetToken<DetGeomDesc, VeryForwardRealGeometryRecord> dgdRealToken_;
  const edm::ESGetToken<DetGeomDesc, VeryForwardMisalignedGeometryRecord> dgdMisToken_;
};

CTPPSGeometryESModule::CTPPSGeometryESModule(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity")),
      fromDD4hep_(iConfig.getUntrackedParameter<bool>("fromDD4hep", false)),
      gdRealTokens_{setWhatProduced(this, &CTPPSGeometryESModule::produceRealGD)},
      gdMisTokens_{setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedGD)},
      dgdRealToken_{
          setWhatProduced(this, &CTPPSGeometryESModule::produceRealTG).consumes<DetGeomDesc>(edm::ESInputTag())},
      dgdMisToken_{
          setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedTG).consumes<DetGeomDesc>(edm::ESInputTag())} {
  auto c = setWhatProduced(this, &CTPPSGeometryESModule::produceIdealGD);

  if (!fromDD4hep_) {
    ddToken_ = c.consumes<DDCompactView>(edm::ESInputTag("", iConfig.getParameter<std::string>("compactViewTag")));
  } else {
    dd4hepToken_ =
        c.consumes<cms::DDCompactView>(edm::ESInputTag("", iConfig.getParameter<std::string>("compactViewTag")));
  }
}

void CTPPSGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 1);
  desc.add<std::string>("compactViewTag", std::string());
  desc.addUntracked<bool>("fromDD4hep", false);
  descriptions.add("CTPPSGeometryESModule", desc);
}

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceIdealGD(const IdealGeometryRecord& iRecord) {
  if (!fromDD4hep_) {
    // Get the DDCompactView from EventSetup
    auto const& myCompactView = iRecord.get(ddToken_);

    // Build geo from compact view.
    return detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView);
  }

  else {
    // Get the DDCompactView from EventSetup
    auto const& myCompactView = iRecord.get(dd4hepToken_);

    // Build geo from compact view.
    return detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView);
  }
}

template <typename REC>
std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceGD(IdealGeometryRecord const& iIdealRec,
                                                              std::optional<REC> const& iAlignRec,
                                                              GDTokens<REC> const& iTokens,
                                                              const char* name) {
  // get the input GeometricalDet
  auto const& idealGD = iIdealRec.get(iTokens.idealGDToken_);

  // load alignments
  CTPPSRPAlignmentCorrectionsData const* alignments = nullptr;
  if (iAlignRec) {
    auto alignmentsHandle = iAlignRec->getHandle(iTokens.alignmentToken_);
    if (alignmentsHandle.isValid()) {
      alignments = alignmentsHandle.product();
    }
  }

  if (verbosity_) {
    if (alignments) {
      edm::LogVerbatim(name) << ">> " << name << " > Real geometry: " << alignments->getRPMap().size() << " RP and "
                             << alignments->getSensorMap().size() << " sensor alignments applied.";
    } else {
      edm::LogVerbatim(name) << ">> " << name << " > Real geometry: No alignment applied.";
    }
  }

  return CTPPSGeometryESCommon::applyAlignments(idealGD, alignments);
}

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceRealGD(const VeryForwardRealGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPRealAlignmentRecord>(),
                   gdRealTokens_,
                   "CTPPSGeometryESModule::produceRealGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceMisalignedGD(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPMisalignedAlignmentRecord>(),
                   gdMisTokens_,
                   "CTPPSGeometryESModule::produceMisalignedGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry> CTPPSGeometryESModule::produceRealTG(const VeryForwardRealGeometryRecord& iRecord) {
  auto const& gD = iRecord.get(dgdRealToken_);

  return std::make_unique<CTPPSGeometry>(&gD, verbosity_);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry> CTPPSGeometryESModule::produceMisalignedTG(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  auto const& gD = iRecord.get(dgdMisToken_);

  return std::make_unique<CTPPSGeometry>(&gD, verbosity_);
}

DEFINE_FWK_EVENTSETUP_MODULE(CTPPSGeometryESModule);
