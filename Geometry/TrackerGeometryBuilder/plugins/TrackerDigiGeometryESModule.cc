#include "TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
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

//__________________________________________________________________
TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet & p) 
{

    applyAlignment_ = p.getUntrackedParameter<bool>("applyAlignment", false);
    fromDDD_ = p.getParameter<bool>("fromDDD");
    if ( fromDDD_ ) {
      setWhatProduced(this, dependsOn( &TrackerDigiGeometryESModule::ddGeometryCallback_ ) );
    } else {
      setWhatProduced(this, dependsOn( &TrackerDigiGeometryESModule::gdGeometryCallback_ ) );
    }
}

//__________________________________________________________________
TrackerDigiGeometryESModule::~TrackerDigiGeometryESModule() {}

//__________________________________________________________________
boost::shared_ptr<TrackerGeometry> 
TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord & iRecord){ 

  //
  // Called whenever the alignments or alignment errors change
  //
  if ( applyAlignment_ ) {
    // Retrieve and apply alignments
    edm::ESHandle<Alignments> alignments;
    iRecord.getRecord<TrackerAlignmentRcd>().get( alignments );
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    iRecord.getRecord<TrackerAlignmentErrorRcd>().get( alignmentErrors );
    GeometryAligner aligner;
    aligner.applyAlignments<TrackerGeometry>( &(*_tracker), &(*alignments), &(*alignmentErrors) );
  }

  return _tracker;

} 


//__________________________________________________________________
void TrackerDigiGeometryESModule::ddGeometryCallback_( const IdealGeometryRecord& record )
{

  //
  // Called whenever the ideal geometry changes
  //
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  record.get( cpv );
  record.get( gD );
  TrackerGeomBuilderFromGeometricDet builder;
  _tracker  = boost::shared_ptr<TrackerGeometry>(builder.build(&(*gD)));

}

void TrackerDigiGeometryESModule::gdGeometryCallback_( const PGeometricDetRcd& record )
{

  //
  // Called whenever the reco-geometric-det-stored-from-DDD geometry changes
  //
  // MEC: blindly copying above example... why they get the compact view without using it is beyond me.
  // MEC: TRY taking out the request for pgd after I get this working.
  edm::ESHandle<PGeometricDet> pgd;
  edm::ESHandle<GeometricDet> gd;
  record.get( pgd );
  record.get( gd );
  TrackerGeomBuilderFromGeometricDet builder;
  _tracker  = boost::shared_ptr<TrackerGeometry>(builder.build(&(*gd)));

}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule);
