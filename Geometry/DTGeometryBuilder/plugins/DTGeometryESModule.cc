/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "DTGeometryESModule.h"
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromCondDB.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include <FWCore/Framework/interface/ESHandle.h>
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

  if(fromDDD_) {
    setWhatProduced(this, dependsOn(&DTGeometryESModule::geometryCallback_) );
  } else {
    setWhatProduced(this, dependsOn(&DTGeometryESModule::dbGeometryCallback_) );
  }

  edm::LogInfo("Geometry") << "@SUB=DTGeometryESModule"
    << "Label '" << myLabel_ << "' "
    << (applyAlignment_ ? "looking for" : "IGNORING")
    << " alignment labels '" << alignmentsLabel_ << "'.";
}

DTGeometryESModule::~DTGeometryESModule(){}

boost::shared_ptr<DTGeometry> 
DTGeometryESModule::produce(const MuonGeometryRecord & record) {

  //
  // Called whenever the alignments or alignment errors change
  //  
  if ( applyAlignment_ ) {
    // applyAlignment_ is scheduled for removal. 
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get(alignmentsLabel_, globalPosition);
    edm::ESHandle<Alignments> alignments;
    record.getRecord<DTAlignmentRcd>().get(alignmentsLabel_, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    record.getRecord<DTAlignmentErrorExtendedRcd>().get(alignmentsLabel_, alignmentErrors);
    // Only apply alignment if values exist
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=DTGeometryRecord::produce"
        << "Alignment(Error)s and global position (label '"
        << alignmentsLabel_ << "') empty: Geometry producer (label "
        << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>( &(*_dtGeometry),
                                           &(*alignments), &(*alignmentErrors),
                                           align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)));
    }
  }

  return _dtGeometry;

}

void DTGeometryESModule::geometryCallback_( const MuonNumberingRecord& record ) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  _dtGeometry = boost::shared_ptr<DTGeometry>(new DTGeometry );
  edm::ESHandle<MuonDDDConstants> mdc;
  record.get( mdc );

  edm::ESTransientHandle<DDCompactView> cpv;
  record.getRecord<IdealGeometryRecord>().get(cpv);

  DTGeometryBuilderFromDDD builder;
  builder.build(_dtGeometry, &(*cpv), *mdc);
    
}

void DTGeometryESModule::dbGeometryCallback_( const DTRecoGeometryRcd& record ) {
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //

  _dtGeometry = boost::shared_ptr<DTGeometry>(new DTGeometry );
  edm::ESHandle<RecoIdealGeometry> rig;
  record.get(rig);
  
  DTGeometryBuilderFromCondDB builder;
  builder.build(_dtGeometry, *rig);

}


DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule);
