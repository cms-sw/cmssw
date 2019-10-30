// -*- C++ -*-
//
// Package:    DetectorDescription/DTGeometryESProducer
// Class:      DTGeometryESProducer
//
/**\class DTGeometryESProducer

 Description: DT Geometry ES producer

 Implementation:
     Based on a copy of original DTGeometryESProducer
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 16 Jan 2019 10:19:37 GMT
//
//
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DTGeometryBuilder.h"

#include <memory>
#include <iostream>
#include <iterator>
#include <string>

using namespace edm;
using namespace std;
using namespace cms;

class DTGeometryESProducer : public ESProducer {
public:
  DTGeometryESProducer(const ParameterSet&);
  ~DTGeometryESProducer() override;

  using ReturnType = shared_ptr<DTGeometry>;
  using Detector = dd4hep::Detector;

  ReturnType produce(const MuonGeometryRecord& record);

private:
  using HostType = ESProductHost<DTGeometry, MuonNumberingRecord, DTRecoGeometryRcd>;

  void setupGeometry(MuonNumberingRecord const&, shared_ptr<HostType>&);
  void setupDBGeometry(DTRecoGeometryRcd const&, shared_ptr<HostType>&);

  ReusableObjectHolder<HostType> m_holder;

  edm::ESGetToken<Alignments, GlobalPositionRcd> m_globalPositionToken;
  edm::ESGetToken<Alignments, DTAlignmentRcd> m_alignmentsToken;
  edm::ESGetToken<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd> m_alignmentErrorsToken;
  edm::ESGetToken<MuonNumbering, MuonNumberingRecord> m_mdcToken;
  edm::ESGetToken<DDDetector, IdealGeometryRecord> m_cpvToken;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> m_registryToken;
  const ESInputTag m_tag;
  const string m_alignmentsLabel;
  const string m_myLabel;
  const string m_attribute;
  const string m_value;
  bool m_applyAlignment;
  bool m_fromDDD;
};

DTGeometryESProducer::DTGeometryESProducer(const ParameterSet& iConfig)
    : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")),
      m_alignmentsLabel(iConfig.getParameter<string>("alignmentsLabel")),
      m_myLabel(iConfig.getParameter<string>("appendToDataLabel")),
      m_attribute(iConfig.getParameter<string>("attribute")),
      m_value(iConfig.getParameter<string>("value")),
      m_fromDDD(iConfig.getParameter<bool>("fromDDD")) {
  m_applyAlignment = iConfig.getParameter<bool>("applyAlignment");

  auto cc = setWhatProduced(this);

  if (m_applyAlignment) {
    m_globalPositionToken = cc.consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{"", m_alignmentsLabel});
    m_alignmentsToken = cc.consumesFrom<Alignments, DTAlignmentRcd>(edm::ESInputTag{"", m_alignmentsLabel});
    m_alignmentErrorsToken =
        cc.consumesFrom<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd>(edm::ESInputTag{"", m_alignmentsLabel});
  }

  if (m_fromDDD) {
    m_mdcToken = cc.consumesFrom<MuonNumbering, MuonNumberingRecord>(edm::ESInputTag{});
    m_cpvToken = cc.consumesFrom<DDDetector, IdealGeometryRecord>(m_tag);
    m_registryToken = cc.consumesFrom<DDSpecParRegistry, DDSpecParRegistryRcd>(m_tag);
  }

  edm::LogInfo("Geometry") << "@SUB=DTGeometryESProducer"
                           << "Label '" << m_myLabel << "' " << (m_applyAlignment ? "looking for" : "IGNORING")
                           << " alignment labels '" << m_alignmentsLabel << "'.";
}

DTGeometryESProducer::~DTGeometryESProducer() {}

std::shared_ptr<DTGeometry> DTGeometryESProducer::produce(const MuonGeometryRecord& record) {
  auto host = m_holder.makeOrGet([]() { return new HostType; });

  {
    BenchmarkGrd counter("DTGeometryESProducer");

    if (m_fromDDD) {
      host->ifRecordChanges<MuonNumberingRecord>(record, [this, &host](auto const& rec) { setupGeometry(rec, host); });
    } else {
      host->ifRecordChanges<DTRecoGeometryRcd>(record, [this, &host](auto const& rec) { setupDBGeometry(rec, host); });
    }
  }
  //
  // Called whenever the alignments or alignment errors change
  //
  if (m_applyAlignment) {
    // m_applyAlignment is scheduled for removal.
    // Ideal geometry obtained by using 'fake alignment' (with m_applyAlignment = true)
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get(m_alignmentsLabel, globalPosition);
    edm::ESHandle<Alignments> alignments;
    record.getRecord<DTAlignmentRcd>().get(m_alignmentsLabel, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    record.getRecord<DTAlignmentErrorExtendedRcd>().get(m_alignmentsLabel, alignmentErrors);
    // Only apply alignment if values exist
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=DTGeometryRecord::produce"
                             << "Alignment(Error)s and global position (label '" << m_alignmentsLabel
                             << "') empty: Geometry producer (label "
                             << "'" << m_myLabel << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>(&(*host),
                                          &(*alignments),
                                          &(*alignmentErrors),
                                          align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)));
    }
  }

  return host;  // automatically converts to std::shared_ptr<DTGeometry>
}

void DTGeometryESProducer::setupGeometry(const MuonNumberingRecord& record, shared_ptr<HostType>& host) {
  host->clear();

  const auto& mdc = record.get(m_mdcToken);

  edm::ESTransientHandle<DDDetector> cpv = record.getTransientHandle(m_cpvToken);

  ESTransientHandle<DDSpecParRegistry> registry = record.getTransientHandle(m_registryToken);

  DDSpecParRefs myReg;
  {
    BenchmarkGrd b1("DTGeometryESProducer Filter Registry");
    registry->filter(myReg, m_attribute, m_value);
  }

  DTGeometryBuilder builder;
  builder.build(*host, cpv.product(), mdc, myReg);
}

void DTGeometryESProducer::setupDBGeometry(const DTRecoGeometryRcd& record, std::shared_ptr<HostType>& host) {
  // host->clear();

  // edm::ESHandle<RecoIdealGeometry> rig;
  // record.get(rig);

  // DTGeometryBuilderFromCondDB builder;
  // builder.build(host, *rig);
}

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESProducer);
