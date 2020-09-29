/** \file
 *
 *  \author N. Amapane - CERN
 */
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDD4Hep.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromCondDB.h"

#include <memory>
#include <iostream>
#include <iterator>
#include <string>

using namespace edm;
using namespace std;

class DTGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  DTGeometryESModule(const edm::ParameterSet& p);

  /// Creation of configuration file
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  /// Produce DTGeometry.
  std::shared_ptr<DTGeometry> produce(const MuonGeometryRecord& record);

private:
  using HostType = edm::ESProductHost<DTGeometry, MuonNumberingRecord, DTRecoGeometryRcd>;

  void setupDDDGeometry(MuonNumberingRecord const&, std::shared_ptr<HostType>&);
  void setupDD4hepGeometry(MuonNumberingRecord const&, std::shared_ptr<HostType>&);
  void setupDBGeometry(DTRecoGeometryRcd const&, std::shared_ptr<HostType>&);

  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, DTAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd> alignmentErrorsToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> mdcToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<cms::DDDetector, IdealGeometryRecord> m_cpvToken;
  edm::ESGetToken<cms::DDSpecParRegistry, DDSpecParRegistryRcd> m_registryToken;
  edm::ESGetToken<RecoIdealGeometry, DTRecoGeometryRcd> rigToken_;

  const edm::ESInputTag m_tag;
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  const std::string m_attribute;
  const std::string m_value;
  bool fromDDD_;
  bool fromDD4hep_;
  bool applyAlignment_;  // Switch to apply alignment corrections
};

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet& p)
    : m_tag(p.getParameter<edm::ESInputTag>("DDDetector")),
      alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
      myLabel_(p.getParameter<std::string>("appendToDataLabel")),
      m_attribute(p.getParameter<std::string>("attribute")),
      m_value(p.getParameter<std::string>("value")),
      fromDDD_(p.getParameter<bool>("fromDDD")),
      fromDD4hep_(p.getParameter<bool>("fromDD4hep")) {
  applyAlignment_ = p.getParameter<bool>("applyAlignment");

  auto cc = setWhatProduced(this);
  if (applyAlignment_) {
    globalPositionToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentsToken_ = cc.consumesFrom<Alignments, DTAlignmentRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentErrorsToken_ =
        cc.consumesFrom<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd>(edm::ESInputTag{"", alignmentsLabel_});
  }
  if (fromDDD_) {
    mdcToken_ = cc.consumesFrom<MuonGeometryConstants, IdealGeometryRecord>(edm::ESInputTag{});
    cpvToken_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{});
  } else if (fromDD4hep_) {
    mdcToken_ = cc.consumesFrom<MuonGeometryConstants, IdealGeometryRecord>(edm::ESInputTag{});
    m_cpvToken = cc.consumesFrom<cms::DDDetector, IdealGeometryRecord>(m_tag);
    m_registryToken = cc.consumesFrom<cms::DDSpecParRegistry, DDSpecParRegistryRcd>(m_tag);
  } else {
    rigToken_ = cc.consumesFrom<RecoIdealGeometry, DTRecoGeometryRcd>(edm::ESInputTag{});
  }

  edm::LogVerbatim("Geometry") << "@SUB=DTGeometryESModule Label '" << myLabel_ << "' "
                               << (applyAlignment_ ? "looking for" : "IGNORING") << " alignment labels '"
                               << alignmentsLabel_ << "'.";
}

void DTGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //
  // This cfi should be included to build the DT geometry model.
  //
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  desc.add<edm::ESInputTag>("DDDetector", edm::ESInputTag("", ""));
  desc.add<std::string>("alignmentsLabel", "");
  desc.add<std::string>("appendToDataLabel", "");
  desc.add<std::string>("attribute", "MuStructure");
  desc.add<std::string>("value", "MuonBarrelDT");
  desc.add<bool>("applyAlignment", true);
  descriptions.add("DTGeometryESModule", desc);
}

std::shared_ptr<DTGeometry> DTGeometryESModule::produce(const MuonGeometryRecord& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  if (fromDDD_) {
    host->ifRecordChanges<MuonNumberingRecord>(record, [this, &host](auto const& rec) { setupDDDGeometry(rec, host); });
  } else if (fromDD4hep_) {
    host->ifRecordChanges<MuonNumberingRecord>(record,
                                               [this, &host](auto const& rec) { setupDD4hepGeometry(rec, host); });
  } else {
    host->ifRecordChanges<DTRecoGeometryRcd>(record, [this, &host](auto const& rec) { setupDBGeometry(rec, host); });
  }
  //
  // Called whenever the alignments or alignment errors change
  //
  if (applyAlignment_) {
    // applyAlignment_ is scheduled for removal.
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    const auto& globalPosition = record.get(globalPositionToken_);
    const auto& alignments = record.get(alignmentsToken_);
    const auto& alignmentErrors = record.get(alignmentErrorsToken_);
    // Only apply alignment if values exist
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogVerbatim("Geometry") << "@SUB=DTGeometryRecord::produce Alignment(Error)s and global position (label '"
                                   << alignmentsLabel_ << "') empty: Geometry producer (label '" << myLabel_
                                   << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>(
          &(*host), &alignments, &alignmentErrors, align::DetectorGlobalPosition(globalPosition, DetId(DetId::Muon)));
    }
  }

  return host;  // automatically converts to std::shared_ptr<DTGeometry>
}

void DTGeometryESModule::setupDDDGeometry(const MuonNumberingRecord& record, std::shared_ptr<HostType>& host) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  host->clear();

  const auto& mdc = record.get(mdcToken_);
  edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(cpvToken_);

  DTGeometryBuilderFromDDD builder;
  builder.build(*host, cpv.product(), mdc);
}

void DTGeometryESModule::setupDD4hepGeometry(const MuonNumberingRecord& record, std::shared_ptr<HostType>& host) {
  host->clear();

  const auto& mdc = record.get(mdcToken_);
  edm::ESTransientHandle<cms::DDDetector> cpv = record.getTransientHandle(m_cpvToken);
  ESTransientHandle<cms::DDSpecParRegistry> registry = record.getTransientHandle(m_registryToken);

  cms::DDSpecParRefs myReg;
  registry->filter(myReg, m_attribute, m_value);

  DTGeometryBuilderFromDD4Hep builder;
  builder.build(*host, cpv.product(), mdc, myReg);
}

void DTGeometryESModule::setupDBGeometry(const DTRecoGeometryRcd& record, std::shared_ptr<HostType>& host) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  host->clear();

  const auto& rig = record.get(rigToken_);

  DTGeometryBuilderFromCondDB builder;
  builder.build(host, rig);
}

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule);
