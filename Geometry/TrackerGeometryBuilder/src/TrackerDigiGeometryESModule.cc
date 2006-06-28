#include "Geometry/TrackerGeometryBuilder/interface/TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

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
    TrackerGeometryAligner aligner;
    aligner.applyAlignments( &(*_tracker), &(*alignments) );
  }

  return _tracker;
}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule)
