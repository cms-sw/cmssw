/****************************************************************************
*  Based on CTPPSGeometryESModule.cc by:
*  Jan Kaspar (jan.kaspar@gmail.com)
*  Dominik Mierzejewski <dmierzej@cern.ch>
*
*  Rewritten + Moved out common functionailities to DetGeomDesc(Builder) by Gabrielle Hugo.
*  Migrated to DD4hep by Wagner Carvalho and Gabrielle Hugo.
*
*  Add the capability of reading PPS reco geometry from the database
*
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardIdealGeometryRecord.h"
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
  std::unique_ptr<DetGeomDesc> produceIdealGDFromPreprocessedDB(const VeryForwardIdealGeometryRecord&);
  std::vector<int> fillCopyNos(TGeoIterator& it);

  std::unique_ptr<DetGeomDesc> produceRealGD(const VeryForwardRealGeometryRecord&);
  std::unique_ptr<DetGeomDesc> produceRealGDFromPreprocessedDB(const VeryForwardRealGeometryRecord&);
  std::unique_ptr<CTPPSGeometry> produceRealTG(const VeryForwardRealGeometryRecord&);

  std::unique_ptr<DetGeomDesc> produceMisalignedGD(const VeryForwardMisalignedGeometryRecord&);
  std::unique_ptr<DetGeomDesc> produceMisalignedGDFromPreprocessedDB(const VeryForwardMisalignedGeometryRecord&);
  std::unique_ptr<CTPPSGeometry> produceMisalignedTG(const VeryForwardMisalignedGeometryRecord&);

  template <typename REC, typename GEO>
  std::unique_ptr<DetGeomDesc> produceGD(const GEO&,
                                         const std::optional<REC>&,
                                         edm::ESGetToken<DetGeomDesc, GEO> const&,
                                         edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, REC> const&,
                                         const char* name);

  const unsigned int verbosity_;
  const bool isRun2_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESGetToken<PDetGeomDesc, VeryForwardIdealGeometryRecord> dbToken_;
  const bool fromPreprocessedDB_, fromDD4hep_;

  edm::ESGetToken<DetGeomDesc, IdealGeometryRecord> idealGDToken_;
  edm::ESGetToken<DetGeomDesc, VeryForwardIdealGeometryRecord> idealDBGDToken_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord> realAlignmentToken_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord> misAlignmentToken_;

  edm::ESGetToken<DetGeomDesc, VeryForwardRealGeometryRecord> dgdRealToken_;
  edm::ESGetToken<DetGeomDesc, VeryForwardMisalignedGeometryRecord> dgdMisToken_;
};

CTPPSGeometryESModule::CTPPSGeometryESModule(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity")),
      isRun2_(iConfig.getParameter<bool>("isRun2")),
      fromPreprocessedDB_(iConfig.getUntrackedParameter<bool>("fromPreprocessedDB", false)),
      fromDD4hep_(iConfig.getUntrackedParameter<bool>("fromDD4hep", false)) {
  if (fromPreprocessedDB_) {
    auto c = setWhatProduced(this, &CTPPSGeometryESModule::produceIdealGDFromPreprocessedDB);
    dbToken_ = c.consumes<PDetGeomDesc>(edm::ESInputTag("", iConfig.getParameter<std::string>("dbTag")));

    auto c1 = setWhatProduced(this, &CTPPSGeometryESModule::produceRealGDFromPreprocessedDB);
    idealDBGDToken_ = c1.consumesFrom<DetGeomDesc, VeryForwardIdealGeometryRecord>(edm::ESInputTag());
    realAlignmentToken_ = c1.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>(edm::ESInputTag());

    auto c2 = setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedGDFromPreprocessedDB);
    misAlignmentToken_ =
        c2.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord>(edm::ESInputTag());
  } else if (!fromDD4hep_) {
    auto c = setWhatProduced(this, &CTPPSGeometryESModule::produceIdealGD);
    ddToken_ = c.consumes<DDCompactView>(edm::ESInputTag("", iConfig.getParameter<std::string>("compactViewTag")));

    auto c1 = setWhatProduced(this, &CTPPSGeometryESModule::produceRealGD);
    idealGDToken_ = c1.consumesFrom<DetGeomDesc, IdealGeometryRecord>(edm::ESInputTag());
    realAlignmentToken_ = c1.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>(edm::ESInputTag());

    auto c2 = setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedGD);
    misAlignmentToken_ =
        c2.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord>(edm::ESInputTag());
  } else {
    auto c = setWhatProduced(this, &CTPPSGeometryESModule::produceIdealGD);
    dd4hepToken_ =
        c.consumes<cms::DDCompactView>(edm::ESInputTag("", iConfig.getParameter<std::string>("compactViewTag")));

    auto c1 = setWhatProduced(this, &CTPPSGeometryESModule::produceRealGD);
    idealGDToken_ = c1.consumesFrom<DetGeomDesc, IdealGeometryRecord>(edm::ESInputTag());
    realAlignmentToken_ = c1.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>(edm::ESInputTag());

    auto c2 = setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedGD);
    misAlignmentToken_ =
        c2.consumesFrom<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord>(edm::ESInputTag());
  }

  auto c_RTG = setWhatProduced(this, &CTPPSGeometryESModule::produceRealTG);
  dgdRealToken_ = c_RTG.consumes<DetGeomDesc>(edm::ESInputTag());

  auto c_MTG = setWhatProduced(this, &CTPPSGeometryESModule::produceMisalignedTG);
  dgdMisToken_ = c_MTG.consumes<DetGeomDesc>(edm::ESInputTag());
}

void CTPPSGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 1);
  desc.add<bool>("isRun2", false)->setComment("Switch to legacy (2017-18) definition of diamond geometry");
  desc.add<std::string>("dbTag", std::string());
  desc.add<std::string>("compactViewTag", std::string());
  desc.addUntracked<bool>("fromPreprocessedDB", false);
  desc.addUntracked<bool>("fromDD4hep", false);
  descriptions.add("CTPPSGeometryESModule", desc);
}

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceIdealGD(const IdealGeometryRecord& iRecord) {
  if (!fromDD4hep_) {
    // Get the DDCompactView from EventSetup
    auto const& myCompactView = iRecord.get(ddToken_);

    // Build geo from compact view.
    return detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView, isRun2_);
  }

  else {
    // Get the DDCompactView from EventSetup
    auto const& myCompactView = iRecord.get(dd4hepToken_);

    // Build geo from compact view.
    return detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView, isRun2_);
  }
}

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceIdealGDFromPreprocessedDB(
    const VeryForwardIdealGeometryRecord& iRecord) {
  // Get the PDetGeomDesc from EventSetup
  auto const& myDB = iRecord.get(dbToken_);

  edm::LogInfo("CTPPSGeometryESModule") << " myDB size = " << myDB.container_.size();

  // Build geo from PDetGeomDesc DB object.
  auto pdet = std::make_unique<DetGeomDesc>(myDB);
  return pdet;
}

template <typename REC, typename GEO>
std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceGD(
    GEO const& iIdealRec,
    std::optional<REC> const& iAlignRec,
    edm::ESGetToken<DetGeomDesc, GEO> const& iGDToken,
    edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, REC> const& iAlignToken,
    const char* name) {
  // get the input GeometricalDet
  auto const& idealGD = iIdealRec.get(iGDToken);

  // load alignments
  CTPPSRPAlignmentCorrectionsData const* alignments = nullptr;
  if (iAlignRec) {
    auto alignmentsHandle = iAlignRec->getHandle(iAlignToken);
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

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceRealGDFromPreprocessedDB(
    const VeryForwardRealGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<VeryForwardIdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPRealAlignmentRecord>(),
                   idealDBGDToken_,
                   realAlignmentToken_,
                   "CTPPSGeometryESModule::produceRealGDFromPreprocessedDB");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceMisalignedGDFromPreprocessedDB(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<VeryForwardIdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPMisalignedAlignmentRecord>(),
                   idealDBGDToken_,
                   misAlignmentToken_,
                   "CTPPSGeometryESModule::produceMisalignedGDFromPreprocessedDB");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceRealGD(const VeryForwardRealGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPRealAlignmentRecord>(),
                   idealGDToken_,
                   realAlignmentToken_,
                   "CTPPSGeometryESModule::produceRealGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> CTPPSGeometryESModule::produceMisalignedGD(
    const VeryForwardMisalignedGeometryRecord& iRecord) {
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(),
                   iRecord.tryToGetRecord<RPMisalignedAlignmentRecord>(),
                   idealGDToken_,
                   misAlignmentToken_,
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
