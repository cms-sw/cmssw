#include "Geometry/TrackerGeometryBuilder/plugins/TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/plugins/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet & p) 
{

    applyAlignment_ = p.getUntrackedParameter<bool>("applyAlignment", false);

    setWhatProduced(this);
}

TrackerDigiGeometryESModule::~TrackerDigiGeometryESModule() {}

boost::shared_ptr<TrackerGeometry> 
TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord & iRecord){ 
  //
  // get the DDCompactView first
  //
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );
  iRecord.getRecord<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet builder;
  _tracker  = boost::shared_ptr<TrackerGeometry>(builder.build(&(*cpv),&(*gD)));

  // Retrieve and apply alignments
  if ( applyAlignment_ ) {
    edm::ESHandle<Alignments> alignments;
    iRecord.getRecord<TrackerAlignmentRcd>().get( alignments );
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    iRecord.getRecord<TrackerAlignmentErrorRcd>().get( alignmentErrors );
    GeometryAligner aligner;
    aligner.applyAlignments<TrackerGeometry>( &(*_tracker), &(*alignments), &(*alignmentErrors) );
  }

  return _tracker;
}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule);
