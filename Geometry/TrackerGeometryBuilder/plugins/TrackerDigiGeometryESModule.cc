#include "TrackerDigiGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>

//__________________________________________________________________
TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet & p) 
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel")),
    m_ROWS_PER_ROC( 80 ),     // Num of Rows per ROC 
    m_COLS_PER_ROC( 52 ),     // Num of Cols per ROC
    m_BIG_PIX_PER_ROC_X( 1 ), // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
    m_BIG_PIX_PER_ROC_Y( 2 ), // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
    m_ROCS_X( 0 ),	      // 2 for SLHC
    m_ROCS_Y( 0 ),	      // 8 for SLHC
    m_upgradeGeometry( false )
{
  if( p.existsAs<edm::ParameterSet>("trackerGeometryConstants"))
  {
      
    const edm::ParameterSet tkGeomConsts(p.getParameter<edm::ParameterSet>("trackerGeometryConstants"));

    m_ROWS_PER_ROC  = tkGeomConsts.getParameter<int>( "ROWS_PER_ROC" );
    m_COLS_PER_ROC  = tkGeomConsts.getParameter<int>( "COLS_PER_ROC" );
    m_BIG_PIX_PER_ROC_X = tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_X" );
    m_BIG_PIX_PER_ROC_Y = tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_Y" );
    m_ROCS_X = tkGeomConsts.getParameter<int>( "ROCS_X" );
    m_ROCS_Y = tkGeomConsts.getParameter<int>( "ROCS_Y" );
    m_upgradeGeometry = tkGeomConsts.getParameter<bool>( "upgradeGeometry" );
  }
     
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
  _tracker  = boost::shared_ptr<TrackerGeometry>(builder.build(&(*gD), m_upgradeGeometry,
							       m_ROWS_PER_ROC,
							       m_COLS_PER_ROC,
							       m_BIG_PIX_PER_ROC_X,
							       m_BIG_PIX_PER_ROC_Y,
							       m_ROCS_X, m_ROCS_Y ));

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
