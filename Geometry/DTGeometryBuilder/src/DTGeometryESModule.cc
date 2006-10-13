/** \file
 *
 *  $Date: 2006/08/22 15:58:37 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/DTGeometryBuilder/src/DTGeometryESModule.h>
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/DataRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <memory>

using namespace edm;

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet & p){

  applyAlignment_ = p.getUntrackedParameter<bool>("applyAlignment", false);

  setWhatProduced(this);
}


DTGeometryESModule::~DTGeometryESModule(){}


boost::shared_ptr<DTGeometry>
DTGeometryESModule::produce(const MuonGeometryRecord & record) {
  edm::ESHandle<DDCompactView> cpv;
  record.getRecord<IdealGeometryRecord>().get(cpv);
  edm::ESHandle<MuonDDDConstants> mdc;
  record.getRecord<MuonNumberingRecord>().get(mdc);
  DTGeometryBuilderFromDDD builder;
  _dtGeometry = boost::shared_ptr<DTGeometry>(builder.build(&(*cpv), *mdc));

  // Retrieve and apply alignments
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

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule)
