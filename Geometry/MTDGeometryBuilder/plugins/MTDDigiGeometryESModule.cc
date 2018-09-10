#include "MTDDigiGeometryESModule.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomBuilderFromGeometricTimingDet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDSurfaceDeformationRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

//__________________________________________________________________
MTDDigiGeometryESModule::MTDDigiGeometryESModule(const edm::ParameterSet & p) 
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{
    applyAlignment_ = p.getParameter<bool>("applyAlignment");
    fromDDD_ = p.getParameter<bool>("fromDDD");

    setWhatProduced(this);

    edm::LogInfo("Geometry") << "@SUB=MTDDigiGeometryESModule"
			     << "Label '" << myLabel_ << "' "
			     << (applyAlignment_ ? "looking for" : "IGNORING")
			     << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
MTDDigiGeometryESModule::~MTDDigiGeometryESModule() {}

void
MTDDigiGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription descDB;
  descDB.add<std::string>( "appendToDataLabel", "" );
  descDB.add<bool>( "fromDDD", false );
  descDB.add<bool>( "applyAlignment", true );
  descDB.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "mtdGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<std::string>( "appendToDataLabel", "" );
  desc.add<bool>( "fromDDD", true );
  desc.add<bool>( "applyAlignment", true );
  desc.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "mtdGeometry", desc );
}

//__________________________________________________________________
std::shared_ptr<MTDGeometry> 
MTDDigiGeometryESModule::produce(const MTDDigiGeometryRecord & iRecord)
{ 
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  edm::ESHandle<GeometricTimingDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( gD );

  edm::ESHandle<MTDTopology> tTopoHand;
  iRecord.getRecord<MTDTopologyRcd>().get(tTopoHand);
  const MTDTopology *tTopo=tTopoHand.product();

  edm::ESHandle<PMTDParameters> ptp;
  iRecord.getRecord<PMTDParametersRcd>().get( ptp );
  
  MTDGeomBuilderFromGeometricTimingDet builder;
  mtd_  = std::shared_ptr<MTDGeometry>(builder.build(&(*gD), *ptp, tTopo));

  
  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    edm::ESHandle<Alignments> globalPosition;
    iRecord.getRecord<GlobalPositionRcd>().get(alignmentsLabel_, globalPosition);
    edm::ESHandle<Alignments> alignments;
    iRecord.getRecord<MTDAlignmentRcd>().get(alignmentsLabel_, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    iRecord.getRecord<MTDAlignmentErrorExtendedRcd>().get(alignmentsLabel_, alignmentErrors);
    // apply if not empty:
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
	 		     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<MTDGeometry>(&(*mtd_), &(*alignments), &(*alignmentErrors),
				       align::DetectorGlobalPosition(*globalPosition,
								     DetId(DetId::Forward)));
    }

    edm::ESHandle<AlignmentSurfaceDeformations> surfaceDeformations;
    iRecord.getRecord<MTDSurfaceDeformationRcd>().get(alignmentsLabel_, surfaceDeformations);
    // apply if not empty:
    if (surfaceDeformations->empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
			     << "AlignmentSurfaceDeformations (label '"
			     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.attachSurfaceDeformations<MTDGeometry>(&(*mtd_), &(*surfaceDeformations));
    }
  }
  
  
  return mtd_;
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDDigiGeometryESModule);
