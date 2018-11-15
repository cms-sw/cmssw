#include "TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

//__________________________________________________________________
TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet & p) 
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{
    applyAlignment_ = p.getParameter<bool>("applyAlignment");
    fromDDD_ = p.getParameter<bool>("fromDDD");

    setWhatProduced(this);

    edm::LogInfo("Geometry") << "@SUB=TrackerDigiGeometryESModule"
			     << "Label '" << myLabel_ << "' "
			     << (applyAlignment_ ? "looking for" : "IGNORING")
			     << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
TrackerDigiGeometryESModule::~TrackerDigiGeometryESModule() {}

void
TrackerDigiGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription descDB;
  descDB.add<std::string>( "appendToDataLabel", "" );
  descDB.add<bool>( "fromDDD", false );
  descDB.add<bool>( "applyAlignment", true );
  descDB.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "trackerGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<std::string>( "appendToDataLabel", "" );
  desc.add<bool>( "fromDDD", true );
  desc.add<bool>( "applyAlignment", true );
  desc.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "trackerGeometry", desc );
}

//__________________________________________________________________
std::unique_ptr<TrackerGeometry> 
TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord & iRecord)
{ 
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  edm::ESHandle<GeometricDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( gD );

  edm::ESHandle<TrackerTopology> tTopoHand;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  edm::ESHandle<PTrackerParameters> ptp;
  iRecord.getRecord<PTrackerParametersRcd>().get( ptp );
  
  TrackerGeomBuilderFromGeometricDet builder;
  std::unique_ptr<TrackerGeometry> tracker(builder.build(&(*gD), *ptp, tTopo));

  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    edm::ESHandle<Alignments> globalPosition;
    iRecord.getRecord<GlobalPositionRcd>().get(alignmentsLabel_, globalPosition);
    edm::ESHandle<Alignments> alignments;
    iRecord.getRecord<TrackerAlignmentRcd>().get(alignmentsLabel_, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    iRecord.getRecord<TrackerAlignmentErrorExtendedRcd>().get(alignmentsLabel_, alignmentErrors);
    // apply if not empty:
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=TrackerDigiGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
	 		     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<TrackerGeometry>(&(*tracker), &(*alignments), &(*alignmentErrors),
                                           align::DetectorGlobalPosition(*globalPosition,
                                                                         DetId(DetId::Tracker)));
    }

    edm::ESHandle<AlignmentSurfaceDeformations> surfaceDeformations;
    iRecord.getRecord<TrackerSurfaceDeformationRcd>().get(alignmentsLabel_, surfaceDeformations);
    // apply if not empty:
    if (surfaceDeformations->empty()) {
      edm::LogInfo("Config") << "@SUB=TrackerDigiGeometryRecord::produce"
			     << "AlignmentSurfaceDeformations (label '"
			     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.attachSurfaceDeformations<TrackerGeometry>(&(*tracker), &(*surfaceDeformations));
    }
  }
  
  return tracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule);
