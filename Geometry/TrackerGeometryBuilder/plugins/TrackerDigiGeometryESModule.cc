#include "TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

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
    myLabel_(p.getParameter<std::string>("appendToDataLabel")),
    m_pSet( p )
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
  edm::ParameterSetDescription descTrackerGeometryConstants;
  descTrackerGeometryConstants.add<bool>( "upgradeGeometry", false );
  descTrackerGeometryConstants.add<int>( "ROWS_PER_ROC", 80 );
  descTrackerGeometryConstants.add<int>( "COLS_PER_ROC", 52 );
  descTrackerGeometryConstants.add<int>( "BIG_PIX_PER_ROC_X", 1 );
  descTrackerGeometryConstants.add<int>( "BIG_PIX_PER_ROC_Y", 2 );
  descTrackerGeometryConstants.add<int>( "ROCS_X", 0 );
  descTrackerGeometryConstants.add<int>( "ROCS_Y", 0 );

  edm::ParameterSetDescription descTrackerSLHCGeometryConstants;
  descTrackerSLHCGeometryConstants.add<bool>( "upgradeGeometry", true );
  descTrackerSLHCGeometryConstants.add<int>( "ROWS_PER_ROC", 80 );
  descTrackerSLHCGeometryConstants.add<int>( "COLS_PER_ROC", 52 );
  descTrackerSLHCGeometryConstants.add<int>( "BIG_PIX_PER_ROC_X", 0 );
  descTrackerSLHCGeometryConstants.add<int>( "BIG_PIX_PER_ROC_Y", 0 );
  descTrackerSLHCGeometryConstants.add<int>( "ROCS_X", 2 );
  descTrackerSLHCGeometryConstants.add<int>( "ROCS_Y", 8 );

  edm::ParameterSetDescription descDB;
  descDB.add<std::string>( "appendToDataLabel", "" );
  descDB.add<bool>( "fromDDD", false );
  descDB.add<bool>( "applyAlignment", true );
  descDB.add<std::string>( "alignmentsLabel", "" );
  descDB.addOptional<edm::ParameterSetDescription>( "trackerGeometryConstants", descTrackerGeometryConstants );
  descriptions.add( "trackerGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<std::string>( "appendToDataLabel", "" );
  desc.add<bool>( "fromDDD", true );
  desc.add<bool>( "applyAlignment", true );
  desc.add<std::string>( "alignmentsLabel", "" );
  desc.addOptional<edm::ParameterSetDescription>( "trackerGeometryConstants", descTrackerGeometryConstants );
  descriptions.add( "trackerGeometry", desc );

  edm::ParameterSetDescription descSLHCDB;
  descSLHCDB.add<std::string>( "appendToDataLabel", "" );
  descSLHCDB.add<bool>( "fromDDD", false );
  descSLHCDB.add<bool>( "applyAlignment", true );
  descSLHCDB.add<std::string>( "alignmentsLabel", "" );
  descSLHCDB.addOptional<edm::ParameterSetDescription>( "trackerGeometryConstants", descTrackerSLHCGeometryConstants );
  descriptions.add( "trackerSLHCGeometryDB", descSLHCDB );

  edm::ParameterSetDescription descSLHC;
  descSLHC.add<std::string>( "appendToDataLabel", "" );
  descSLHC.add<bool>( "fromDDD", true );
  descSLHC.add<bool>( "applyAlignment", true );
  descSLHC.add<std::string>( "alignmentsLabel", "" );
  descSLHC.addOptional<edm::ParameterSetDescription>( "trackerGeometryConstants", descTrackerSLHCGeometryConstants );
  descriptions.add( "trackerSLHCGeometry", descSLHC );
}

//__________________________________________________________________
boost::shared_ptr<TrackerGeometry> 
TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord & iRecord)
{ 
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  edm::ESHandle<GeometricDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( gD );
  
  TrackerGeomBuilderFromGeometricDet builder;
  _tracker  = boost::shared_ptr<TrackerGeometry>(builder.build(&(*gD), m_pSet ));

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
      ali.applyAlignments<TrackerGeometry>(&(*_tracker), &(*alignments), &(*alignmentErrors),
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
      ali.attachSurfaceDeformations<TrackerGeometry>(&(*_tracker), &(*surfaceDeformations));
    }
  }
  
  return _tracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule);
