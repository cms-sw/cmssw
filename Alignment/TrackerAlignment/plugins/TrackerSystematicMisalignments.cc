#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CLHEP/Random/RandGauss.h"

#include "Alignment/TrackerAlignment/plugins/TrackerSystematicMisalignments.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

TrackerSystematicMisalignments::TrackerSystematicMisalignments(const edm::ParameterSet& cfg)
{
	// use existing geometry
	m_fromDBGeom = cfg.getUntrackedParameter< bool > ("fromDBGeom");
	
	// flags
	m_radialFlag = cfg.getUntrackedParameter< bool > ("radialFlag");
	m_telescopeFlag = cfg.getUntrackedParameter< bool > ("telescopeFlag");
	m_layerRotFlag = cfg.getUntrackedParameter< bool > ("layerRotFlag");
	m_bowingFlag = cfg.getUntrackedParameter< bool > ("bowingFlag");
	m_zExpFlag = cfg.getUntrackedParameter< bool > ("zExpFlag");
	m_twistFlag = cfg.getUntrackedParameter< bool > ("twistFlag");
	m_ellipticalFlag = cfg.getUntrackedParameter< bool > ("ellipticalFlag");
	m_skewFlag = cfg.getUntrackedParameter< bool > ("skewFlag");
	m_saggitaFlag = cfg.getUntrackedParameter< bool > ("saggitaFlag");
	
	// constants
	m_radialEpsilon = cfg.getUntrackedParameter< double > ("radialEpsilon");
	m_telescopeEpsilon = cfg.getUntrackedParameter< double > ("telescopeEpsilon");
	m_layerRotEpsilon = cfg.getUntrackedParameter< double > ("layerRotEpsilon");
	m_bowingEpsilon = cfg.getUntrackedParameter< double > ("bowingEpsilon");
	m_zExpEpsilon = cfg.getUntrackedParameter< double > ("zExpEpsilon");
	m_twistEpsilon = cfg.getUntrackedParameter< double > ("twistEpsilon");
	m_ellipticalEpsilon = cfg.getUntrackedParameter< double > ("ellipticalEpsilon");
	m_skewEpsilon = cfg.getUntrackedParameter< double > ("skewEpsilon");
	m_saggitaEpsilon = cfg.getUntrackedParameter< double > ("saggitaEpsilon");
	
	if (m_radialFlag){
		edm::LogWarning("MisalignedTracker") << "Applying radial ...";		
	}
	if (m_telescopeFlag){
		edm::LogWarning("MisalignedTracker") << "Applying telescope ...";		
	}
	if (m_layerRotFlag){
		edm::LogWarning("MisalignedTracker") << "Applying layer rotation ...";		
	}
	if (m_bowingFlag){
		edm::LogWarning("MisalignedTracker") << "Applying bowing ...";		
	}
	if (m_zExpFlag){
		edm::LogWarning("MisalignedTracker") << "Applying z-expansion ...";		
	}
	if (m_twistFlag){
		edm::LogWarning("MisalignedTracker") << "Applying twist ...";		
	}
	if (m_ellipticalFlag){
		edm::LogWarning("MisalignedTracker") << "Applying elliptical ...";		
	}
	if (m_skewFlag){
		edm::LogWarning("MisalignedTracker") << "Applying skew ...";		
	}
	if (m_saggitaFlag){
		edm::LogWarning("MisalignedTracker") << "Applying saggita ...";		
	}
	
}

void TrackerSystematicMisalignments::beginJob(const edm::EventSetup& setup)
{
	std::cout << "IN BEGIN JOB" << std::endl;
	
	
}


void TrackerSystematicMisalignments::analyze(const edm::Event& event, const edm::EventSetup& setup){
	
	std::cout << "IN ANALYZE" << std::endl;
	
	edm::ESHandle<GeometricDet>  geom;
	setup.get<IdealGeometryRecord>().get(geom);	 
	//edm::ESHandle<DDCompactView> cpv;
	//setup.get<IdealGeometryRecord>().get(cpv);
	TrackerGeometry* tracker = TrackerGeomBuilderFromGeometricDet().build(&*geom);
	
	//take geometry from DB or randomly generate geometry
	if (m_fromDBGeom){
		//build the tracker
		edm::ESHandle<Alignments> alignments;
		edm::ESHandle<AlignmentErrors> alignmentErrors;
		
		setup.get<TrackerAlignmentRcd>().get(alignments);
		setup.get<TrackerAlignmentErrorRcd>().get(alignmentErrors);
		
		//apply the latest alignments
		GeometryAligner aligner;
		aligner.applyAlignments<TrackerGeometry>( &(*tracker), &(*alignments), &(*alignmentErrors), AlignTransform() );
		
	}
	
	theAlignableTracker = new AlignableTracker(&(*tracker));
	
	applySystematicMisalignment( &(*theAlignableTracker) );
	
	
	// -------------- writing out to alignment record --------------
	///*
	Alignments* myAlignments = theAlignableTracker->alignments() ;
	AlignmentErrors* myAlignmentErrors = theAlignableTracker->alignmentErrors() ;
	
	
	// 2. Store alignment[Error]s to DB
	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	std::string theAlignRecordName = "TrackerAlignmentRcd";
	std::string theErrorRecordName = "TrackerAlignmentErrorRcd";
	
	// 2. Store alignment[Error]s to DB
	// Call service
	if( !poolDbService.isAvailable() ) // Die if not available
		throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
	
	poolDbService->writeOne<Alignments>(&(*myAlignments), poolDbService->beginOfTime(), theAlignRecordName);
	poolDbService->writeOne<AlignmentErrors>(&(*myAlignmentErrors), poolDbService->beginOfTime(), theErrorRecordName);
	//*/
	
}

void TrackerSystematicMisalignments::applySystematicMisalignment(Alignable* ali)
{
	
	const align::Alignables& comp = ali->components();
	unsigned int nComp = comp.size();
	//move then do for lower level object
	//for issue of det vs detunit
	bool usecomps = true;
	if ((ali->alignableObjectId()==2)&&(nComp>=1)) usecomps = false;
	for (unsigned int i = 0; i < nComp; ++i){
		if (usecomps) applySystematicMisalignment(comp[i]);
	}
	DetId id( ali->id() );
	//int subdetlevel = id.subdetId();
	int level = ali->alignableObjectId();
	
	if ((level == 1)||(level == 2)){
		
		align::PositionType gP = ali->globalPosition();
		align::GlobalVector gVec = findSystematicMis( gP );
		ali->move( gVec );
	}
	
}	 

align::GlobalVector TrackerSystematicMisalignments::findSystematicMis( align::PositionType globalPos ){
	
	double newX = 0.0;
	double newY = 0.0;
	double newZ = 0.0;
	double oldX = globalPos.x();
	double oldY = globalPos.y();
	double oldZ = globalPos.z();
	double oldPhi = globalPos.phi();
	double oldR = sqrt(globalPos.x()*globalPos.x() + globalPos.y()*globalPos.y());
	
	if (m_radialFlag){
		newX += m_radialEpsilon*oldX;
		newY += m_radialEpsilon*oldY;
	}
	if (m_telescopeFlag){
		newZ += m_telescopeEpsilon*oldR;
	}
	if (m_layerRotFlag){
		double xP = oldR*cos(oldPhi+m_layerRotEpsilon*(oldR-57.0));
		double yP = oldR*sin(oldPhi+m_layerRotEpsilon*(oldR-57.0));
		newX += (xP - oldX);
		newY += (yP - oldY);
	}
	if (m_bowingFlag){
		newX += oldX*m_bowingEpsilon*(271.846*271.846-oldZ*oldZ);
		newY += oldY*m_bowingEpsilon*(271.846*271.846-oldZ*oldZ);
	}
	if (m_zExpFlag){
		newZ += oldZ*m_zExpEpsilon;
	}
	if (m_twistFlag){
		double xP = oldR*cos(oldPhi+m_twistEpsilon*oldZ);
		double yP = oldR*sin(oldPhi+m_twistEpsilon*oldZ);
		newX += (xP - oldX);
		newY += (yP - oldY);
	}
	if (m_ellipticalFlag){
		newX += oldX*m_ellipticalEpsilon*cos(2.0*oldPhi);
		newY += oldY*m_ellipticalEpsilon*cos(2.0*oldPhi);
	}
	if (m_skewFlag){
		newZ += m_skewEpsilon*cos(oldPhi);
	}
	if (m_saggitaFlag){
		// newX += oldX/fabs(oldX)*m_saggitaEpsilon; // old one...
		newY += oldR*m_saggitaEpsilon;
	}
	
	// strange convention for global z
	align::GlobalVector gV( newX, newY, (-1)*newZ );
	return gV;
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackerSystematicMisalignments);
