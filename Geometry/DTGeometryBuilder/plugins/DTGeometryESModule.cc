/** \file
 *
 *  $Date: 2008/02/22 12:49:23 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN
 */

#include "DTGeometryESModule.h"
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <memory>

using namespace edm;

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet & p)
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{

  applyAlignment_ = p.getParameter<bool>("applyAlignment");

  setWhatProduced(this, dependsOn(&DTGeometryESModule::geometryCallback_) );

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
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    record.getRecord<DTAlignmentErrorRcd>().get(alignmentsLabel_, alignmentErrors);
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


//______________________________________________________________________________
void DTGeometryESModule::geometryCallback_( const MuonNumberingRecord& record )
{
  
  //
  // Called whenever the muon numbering (or ideal geometry) changes
  //
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<MuonDDDConstants> mdc;
  record.getRecord<IdealGeometryRecord>().get(cpv);
  record.get( mdc );
  DTGeometryBuilderFromDDD builder;
  _dtGeometry = boost::shared_ptr<DTGeometry>(builder.build(&(*cpv), *mdc));

}
DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule);
