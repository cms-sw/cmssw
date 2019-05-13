/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "DTGeometryESModule.h"
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromCondDB.h>

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include <FWCore/Framework/interface/ESTransientHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <memory>
#include <iostream>

using namespace edm;
using namespace std;

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet & p)
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel")),
    fromDDD_(p.getParameter<bool>("fromDDD"))
{

  applyAlignment_ = p.getParameter<bool>("applyAlignment");

  auto cc = setWhatProduced(this);
  if(applyAlignment_) {
    globalPositionToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentsToken_ = cc.consumesFrom<Alignments, DTAlignmentRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentErrorsToken_ = cc.consumesFrom<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd>(edm::ESInputTag{"", alignmentsLabel_});
  }
  if(fromDDD_) {
    mdcToken_ = cc.consumesFrom<MuonDDDConstants, MuonNumberingRecord>(edm::ESInputTag{});
    cpvToken_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{});
  }
  else {
    rigToken_ = cc.consumesFrom<RecoIdealGeometry, DTRecoGeometryRcd>(edm::ESInputTag{});
  }

  edm::LogInfo("Geometry") << "@SUB=DTGeometryESModule"
    << "Label '" << myLabel_ << "' "
    << (applyAlignment_ ? "looking for" : "IGNORING")
    << " alignment labels '" << alignmentsLabel_ << "'.";
}

DTGeometryESModule::~DTGeometryESModule(){}

std::shared_ptr<DTGeometry> 
DTGeometryESModule::produce(const MuonGeometryRecord & record) {

  auto host = holder_.makeOrGet([]() {
    return new HostType;
  });

  if(fromDDD_) {
    host->ifRecordChanges<MuonNumberingRecord>(record,
                                               [this, &host](auto const& rec) {
      setupGeometry(rec, host);
    });
  } else {
    host->ifRecordChanges<DTRecoGeometryRcd>(record,
                                             [this, &host](auto const& rec) {
      setupDBGeometry(rec, host);
    });
  }
  //
  // Called whenever the alignments or alignment errors change
  //  
  if ( applyAlignment_ ) {
    // applyAlignment_ is scheduled for removal. 
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    const auto& globalPosition = record.get(globalPositionToken_);
    const auto& alignments = record.get(alignmentsToken_);
    const auto& alignmentErrors = record.get(alignmentErrorsToken_);
    // Only apply alignment if values exist
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogInfo("Config") << "@SUB=DTGeometryRecord::produce"
        << "Alignment(Error)s and global position (label '"
        << alignmentsLabel_ << "') empty: Geometry producer (label "
        << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>( &(*host),
                                           &alignments, &alignmentErrors,
                                           align::DetectorGlobalPosition(globalPosition, DetId(DetId::Muon)));
    }
  }

  return host; // automatically converts to std::shared_ptr<DTGeometry>

}

void DTGeometryESModule::setupGeometry( const MuonNumberingRecord& record,
                                        std::shared_ptr<HostType>& host) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  host->clear();

  const auto& mdc = record.get(mdcToken_);
  edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(cpvToken_);

  DTGeometryBuilderFromDDD builder;
  builder.build(*host, cpv.product(), mdc);
}

void DTGeometryESModule::setupDBGeometry( const DTRecoGeometryRcd& record,
                                          std::shared_ptr<HostType>& host ) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  host->clear();

  const auto& rig = record.get(rigToken_);
  
  DTGeometryBuilderFromCondDB builder;
  builder.build(host, rig);
}

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule);
