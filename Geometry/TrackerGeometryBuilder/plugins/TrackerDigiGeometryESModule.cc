#include "TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

//__________________________________________________________________
TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet & p) 
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{

    applyAlignment_ = p.getParameter<bool>("applyAlignment");
    fromDDD_ = p.getParameter<bool>("fromDDD");
    if ( fromDDD_ ) {
      setWhatProduced(this, dependsOn( &TrackerDigiGeometryESModule::ddGeometryCallback_ ) );
    } else {
      setWhatProduced(this, dependsOn( &TrackerDigiGeometryESModule::gdGeometryCallback_ ) );
    }

    edm::LogInfo("Geometry") << "@SUB=TrackerDigiGeometryESModule"
			     << "Label '" << myLabel_ << "' "
			     << (applyAlignment_ ? "looking for" : "IGNORING")
			     << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
TrackerDigiGeometryESModule::~TrackerDigiGeometryESModule() {}

//__________________________________________________________________
boost::shared_ptr<TrackerGeometry> 
TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord & iRecord)
{ 

  //
  // Called whenever the alignments, alignment errors or global positions change
  //

  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    edm::ESHandle<Alignments> globalPosition;
    iRecord.getRecord<GlobalPositionRcd>().get(alignmentsLabel_, globalPosition);
    edm::ESHandle<Alignments> alignments;
    iRecord.getRecord<TrackerAlignmentRcd>().get(alignmentsLabel_, alignments);
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    iRecord.getRecord<TrackerAlignmentErrorRcd>().get(alignmentsLabel_, alignmentErrors);
    // apply if not empty:
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=TrackerDigiGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
			     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<TrackerGeometry>(&(*_tracker), &(*alignments), &(*alignmentErrors),
                                           align::DetectorGlobalPosition(*globalPosition,
                                                                         DetId(DetId::Tracker)));
    }
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
