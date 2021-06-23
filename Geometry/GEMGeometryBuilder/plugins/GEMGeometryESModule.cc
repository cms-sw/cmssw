/*
//\class GEMGeometryESModule

 Description: GEM Geometry ES Module from DD & DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  27 Jan 2020 
*/
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilder.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include <memory>

class GEMGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  GEMGeometryESModule(const edm::ParameterSet& p);

  /// Define the cfi file
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  /// Produce GEMGeometry.
  std::unique_ptr<GEMGeometry> produce(const MuonGeometryRecord& record);

private:
  // use the DDD as Geometry source
  const bool fromDDD_;
  const bool fromDD4hep_;
  bool applyAlignment_;
  const std::string alignmentsLabel_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> mdcToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepcpvToken_;
  edm::ESGetToken<RecoIdealGeometry, GEMRecoGeometryRcd> riggemToken_;
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, GEMAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, GEMAlignmentErrorExtendedRcd> alignmentErrorsToken_;
};

GEMGeometryESModule::GEMGeometryESModule(const edm::ParameterSet& p)
    : fromDDD_{p.getParameter<bool>("fromDDD")},
      fromDD4hep_{p.getParameter<bool>("fromDD4Hep")},
      applyAlignment_(p.getParameter<bool>("applyAlignment")),
      alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")) {
  auto cc = setWhatProduced(this);
  if (fromDDD_) {
    cpvToken_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else if (fromDD4hep_) {
    dd4hepcpvToken_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else {
    riggemToken_ = cc.consumes();
  }
  if (applyAlignment_) {
    globalPositionToken_ = cc.consumes(edm::ESInputTag{"", alignmentsLabel_});
    alignmentsToken_ = cc.consumes(edm::ESInputTag{"", alignmentsLabel_});
    alignmentErrorsToken_ = cc.consumes(edm::ESInputTag{"", alignmentsLabel_});
  }
  edm::LogVerbatim("GEMGeometry") << "GEMGeometryESModule::initailized with flags " << fromDDD_ << ":" << fromDD4hep_;
}

void GEMGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4Hep", false);
  desc.add<bool>("applyAlignment", false);
  desc.add<std::string>("alignmentsLabel", "");
  descriptions.add("gemGeometry", desc);
}

std::unique_ptr<GEMGeometry> GEMGeometryESModule::produce(const MuonGeometryRecord& record) {
  auto gemGeometry = std::make_unique<GEMGeometry>();

  if (fromDDD_) {
    edm::LogVerbatim("GEMGeometry") << "GEMGeometryESModule::produce :: GEMGeometryBuilder builder ddd";
    auto cpv = record.getTransientHandle(cpvToken_);
    const auto& mdc = record.get(mdcToken_);
    GEMGeometryBuilder builder;
    builder.build(*gemGeometry, cpv.product(), mdc);
  } else if (fromDD4hep_) {
    edm::LogVerbatim("GEMGeometry") << "GEMGeometryESModule::produce :: GEMGeometryBuilder builder dd4hep";
    edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(dd4hepcpvToken_);
    const auto& mdc = record.get(mdcToken_);
    GEMGeometryBuilder builder;
    builder.build(*gemGeometry, cpv.product(), mdc);
  } else {
    edm::LogVerbatim("GEMGeometry") << "GEMGeometryESModule::produce :: GEMGeometryBuilder builder db";
    const auto& riggem = record.get(riggemToken_);
    GEMGeometryBuilderFromCondDB builder;
    builder.build(*gemGeometry, riggem);
  }

  if (applyAlignment_) {
    const auto& globalPosition = record.get(globalPositionToken_);
    const auto& alignments = record.get(alignmentsToken_);
    const auto& alignmentErrors = record.get(alignmentErrorsToken_);

    // No alignment records, assume ideal geometry is wanted
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogInfo("Config") << "@SUB=GEMGeometryRecord::produce"
                             << "Alignment(Error)s and global position (label '" << alignmentsLabel_
                             << "') empty: it is assumed fake and will not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<GEMGeometry>(gemGeometry.get(),
                                           &alignments,
                                           &alignmentErrors,
                                           align::DetectorGlobalPosition(globalPosition, DetId(DetId::Muon)));
    }
  }

  return gemGeometry;
}

DEFINE_FWK_EVENTSETUP_MODULE(GEMGeometryESModule);
