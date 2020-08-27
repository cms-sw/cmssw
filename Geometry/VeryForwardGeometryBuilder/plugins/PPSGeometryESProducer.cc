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

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

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

PPSGeometryESProducer::PPSGeometryESProducer(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity")),
      detectorToken_{setWhatProduced(this, &PPSGeometryESProducer::produceIdealGD)
                         .consumes<cms::DDDetector>(edm::ESInputTag("" /*optional module label */,
                                                                    iConfig.getParameter<std::string>("detectorTag")))},
      gdRealTokens_{setWhatProduced(this, &PPSGeometryESProducer::produceRealGD)},
      gdMisTokens_{setWhatProduced(this, &PPSGeometryESProducer::produceMisalignedGD)},
      dgdRealToken_{
          setWhatProduced(this, &PPSGeometryESProducer::produceRealTG).consumes<DetGeomDesc>(edm::ESInputTag())},
      dgdMisToken_{
          setWhatProduced(this, &PPSGeometryESProducer::produceMisalignedTG).consumes<DetGeomDesc>(edm::ESInputTag())} {
}

void PPSGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 1);
  desc.add<std::string>("detectorTag", std::string());
  descriptions.add("PPSGeometryESProducer", desc);
}

//----------------------------------------------------------------------------------------------------
/*
 * Apply alignments by doing a BFS on idealGD tree.
 */
void PPSGeometryESProducer::applyAlignments(const DetGeomDesc& idealGD,
                                            const CTPPSRPAlignmentCorrectionsData* alignments,
                                            DetGeomDesc*& newGD) {
  newGD = new DetGeomDesc(idealGD);
  std::deque<const DetGeomDesc*> buffer;
  std::deque<DetGeomDesc*> bufferNew;
  buffer.emplace_back(&idealGD);
  bufferNew.emplace_back(newGD);

  while (!buffer.empty()) {
    const DetGeomDesc* sD = buffer.front();
    DetGeomDesc* pD = bufferNew.front();
    buffer.pop_front();
    bufferNew.pop_front();

    const std::string name = pD->name();

    // Is it sensor? If yes, apply full sensor alignments
    if (name == DDD_TOTEM_RP_SENSOR_NAME || name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME ||
        name == DDD_CTPPS_UFSD_SEGMENT_NAME || name == DDD_CTPPS_PIXELS_SENSOR_NAME ||
        std::regex_match(name, std::regex(DDD_TOTEM_TIMING_SENSOR_TMPL))) {
      unsigned int plId = pD->geographicalID();

      if (alignments) {
        const auto& ac = alignments->getFullSensorCorrection(plId);
        pD->applyAlignment(ac);
      }
    }

    // Is it RP box? If yes, apply RP alignments
    if (name == DDD_TOTEM_RP_RP_NAME || name == DDD_CTPPS_DIAMONDS_RP_NAME || name == DDD_CTPPS_PIXELS_RP_NAME ||
        name == DDD_TOTEM_TIMING_RP_NAME) {
      unsigned int rpId = pD->geographicalID();

      if (alignments) {
        const auto& ac = alignments->getRPCorrection(rpId);
        pD->applyAlignment(ac);
      }
    }

    // create and add children
    for (unsigned int i = 0; i < sD->components().size(); i++) {
      const DetGeomDesc* sDC = sD->components()[i];
      buffer.emplace_back(sDC);

      // create new node with the same information as in sDC and add it as a child of pD
      DetGeomDesc* cD = new DetGeomDesc(*sDC);
      pD->addComponent(cD);

      bufferNew.emplace_back(cD);
    }
  }
}

std::unique_ptr<DetGeomDesc> PPSGeometryESProducer::produceIdealGD(const IdealGeometryRecord& iRecord) {
  // Get the DDDetector from EventSetup
  auto const& det = iRecord.get(detectorToken_);

  // Get the DDCompactView
  cms::DDCompactView myCompactView(det);

  // Build geo from compact view.
  return detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView);
}

template <typename REC>
std::unique_ptr<DetGeomDesc> PPSGeometryESProducer::produceGD(IdealGeometryRecord const& iIdealRec,
                                                              std::optional<REC> const& iAlignRec,
                                                              GDTokens<REC> const& iTokens,
                                                              const char* name) {
  // get the input GeometricalDet
  auto const& idealGD = iIdealRec.get(iTokens.idealGDToken_);

  // load alignments
  edm::ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;
  if (iAlignRec) {
    alignments = iAlignRec->getHandle(iTokens.alignmentToken_);
  }

  if (alignments.isValid()) {
    if (verbosity_)
      edm::LogVerbatim(name) << ">> " << name << " > Real geometry: " << alignments->getRPMap().size() << " RP and "
                             << alignments->getSensorMap().size() << " sensor alignments applied.";
  } else {
    if (verbosity_)
      edm::LogVerbatim(name) << ">> " << name << " > Real geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = nullptr;
  applyAlignments(idealGD, alignments.product(), newGD);
  return std::unique_ptr<DetGeomDesc>(newGD);
}

std::unique_ptr<DetGeomDesc> PPSGeometryESProducer::produceRealGD(const VeryForwardRealGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPRealAlignmentRecord>(),
                   gdRealTokens_,
                   "PPSGeometryESProducer::produceRealGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> PPSGeometryESProducer::produceMisalignedGD(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPMisalignedAlignmentRecord>(),
                   gdMisTokens_,
                   "PPSGeometryESProducer::produceMisalignedGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry> PPSGeometryESProducer::produceRealTG(const VeryForwardRealGeometryRecord& iRecord) {
  auto const& gD = iRecord.get(dgdRealToken_);

  return std::make_unique<CTPPSGeometry>(&gD, verbosity_);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry> PPSGeometryESProducer::produceMisalignedTG(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  auto const& gD = iRecord.get(dgdMisToken_);

  return std::make_unique<CTPPSGeometry>(&gD, verbosity_);
}

DEFINE_FWK_EVENTSETUP_MODULE(PPSGeometryESProducer);
