/** \file
 *
 *  $Date: 2007/10/18 08:48:42 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "DTGeometryESModule.h"
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>

// Alignments
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <memory>

using namespace edm;

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet & p){

  applyAlignment_ = p.getUntrackedParameter<bool>("applyAlignment", false);

  setWhatProduced(this, dependsOn(&DTGeometryESModule::geometryCallback_) );
}


DTGeometryESModule::~DTGeometryESModule(){}


boost::shared_ptr<DTGeometry>
DTGeometryESModule::produce(const MuonGeometryRecord & record) {

  //
  // Called whenever the alignments or alignment errors change
  //  
  if ( applyAlignment_ ) {
    edm::ESHandle<Alignments> alignments;
    record.getRecord<DTAlignmentRcd>().get( alignments );
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    record.getRecord<DTAlignmentErrorRcd>().get( alignmentErrors );
    GeometryAligner aligner;
    aligner.applyAlignments<DTGeometry>( &(*_dtGeometry),
                                         &(*alignments), &(*alignmentErrors) );
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
