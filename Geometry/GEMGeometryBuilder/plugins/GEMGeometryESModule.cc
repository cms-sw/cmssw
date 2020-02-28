/*
//\class GEMGeometryESModule

 Description: GEM Geometry ES Module from DD & DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  27 Jan 2020 
*/
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include <memory>

//dd4hep
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

using namespace edm;

class GEMGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  GEMGeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  ~GEMGeometryESModule() override;

  /// Produce GEMGeometry.
  std::unique_ptr<GEMGeometry> produce(const MuonGeometryRecord& record);

private:
  // use the DDD as Geometry source
  const bool useDDD_;
  const bool useDD4hep_;
  bool applyAlignment_;
  const std::string alignmentsLabel_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> mdcToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepcpvToken_;
  edm::ESGetToken<cms::MuonNumbering, MuonNumberingRecord> dd4hepmdcToken_;
  edm::ESGetToken<RecoIdealGeometry, GEMRecoGeometryRcd> riggemToken_;
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, GEMAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, GEMAlignmentErrorExtendedRcd> alignmentErrorsToken_;
};

GEMGeometryESModule::GEMGeometryESModule(const edm::ParameterSet& p)
    : useDDD_{p.getParameter<bool>("useDDD")},
      useDD4hep_{p.getUntrackedParameter<bool>("useDD4hep", false)},
      applyAlignment_(p.getParameter<bool>("applyAlignment")),
      alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")) {
  auto cc = setWhatProduced(this);
  if (useDDD_) {
    cc.setConsumes(cpvToken_).setConsumes(mdcToken_);
  } else if (useDD4hep_) {
    cc.setConsumes(dd4hepcpvToken_).setConsumes(dd4hepmdcToken_);
  } else {
    cc.setConsumes(riggemToken_);
  }
  if (applyAlignment_) {
    cc.setConsumes(globalPositionToken_, edm::ESInputTag{"", alignmentsLabel_})
        .setConsumes(alignmentsToken_, edm::ESInputTag{"", alignmentsLabel_})
        .setConsumes(alignmentErrorsToken_, edm::ESInputTag{"", alignmentsLabel_});
  }
}

GEMGeometryESModule::~GEMGeometryESModule() {}

std::unique_ptr<GEMGeometry> GEMGeometryESModule::produce(const MuonGeometryRecord& record) {
  auto gemGeometry = std::make_unique<GEMGeometry>();

  if (useDDD_) {
    auto cpv = record.getTransientHandle(cpvToken_);
    const auto& mdc = record.get(mdcToken_);
    GEMGeometryBuilderFromDDD builder;
    builder.build(*gemGeometry, cpv.product(), mdc);
  } else if (useDD4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(dd4hepcpvToken_);
    const auto& mdc = record.get(dd4hepmdcToken_);
    GEMGeometryBuilderFromDDD builder;
    builder.build(*gemGeometry, cpv.product(), mdc);
  } else {
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
