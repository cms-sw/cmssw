#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "CLHEP/Random/RandGauss.h"

#include "Alignment/TrackerAlignment/plugins/TrackerSystematicMisalignments.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"  // for enums TID/TIB/etc.

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// -----------------------------------------------------------------
// 2010-05-20 Frank Meier
// Changed sign of z-correction, i.e. z-expansion is now an expansion
// made some variables constant, removed obviously dead code and comments

TrackerSystematicMisalignments::TrackerSystematicMisalignments(const edm::ParameterSet& cfg)
  : theAlignableTracker(nullptr)
{
	// use existing geometry
	m_fromDBGeom = cfg.getUntrackedParameter< bool > ("fromDBGeom");

	// constants
	m_radialEpsilon = cfg.getUntrackedParameter< double > ("radialEpsilon");
	m_telescopeEpsilon = cfg.getUntrackedParameter< double > ("telescopeEpsilon");
	m_layerRotEpsilon = cfg.getUntrackedParameter< double > ("layerRotEpsilon");
	m_bowingEpsilon = cfg.getUntrackedParameter< double > ("bowingEpsilon");
	m_zExpEpsilon = cfg.getUntrackedParameter< double > ("zExpEpsilon");
	m_twistEpsilon = cfg.getUntrackedParameter< double > ("twistEpsilon");
	m_ellipticalEpsilon = cfg.getUntrackedParameter< double > ("ellipticalEpsilon");
	m_skewEpsilon = cfg.getUntrackedParameter< double > ("skewEpsilon");
	m_sagittaEpsilon = cfg.getUntrackedParameter< double > ("sagittaEpsilon");

	m_ellipticalDelta = cfg.getUntrackedParameter< double > ("ellipticalDelta");
	m_skewDelta = cfg.getUntrackedParameter< double > ("skewDelta");
	m_sagittaDelta = cfg.getUntrackedParameter< double > ("sagittaDelta");

	if (m_radialEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying radial ...";
	}
	if (m_telescopeEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying telescope ...";
	}
	if (m_layerRotEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying layer rotation ...";
	}
	if (m_bowingEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying bowing ...";
	}
	if (m_zExpEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying z-expansion ...";
	}
	if (m_twistEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying twist ...";
	}
	if (m_ellipticalEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying elliptical ...";
	}
	if (m_skewEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying skew ...";
	}
	if (m_sagittaEpsilon > -990.0){
		edm::LogWarning("MisalignedTracker") << "Applying sagitta ...";
	}

	// get flag for suppression of blind movements
	suppressBlindMvmts = cfg.getUntrackedParameter< bool > ("suppressBlindMvmts");
	if (suppressBlindMvmts)
	{
		edm::LogWarning("MisalignedTracker") << "Blind movements suppressed (TIB/TOB in z, TID/TEC in r)";
	}

	// compatibility with old (weird) z convention
	oldMinusZconvention = cfg.getUntrackedParameter< bool > ("oldMinusZconvention");
	if (oldMinusZconvention)
	{
		edm::LogWarning("MisalignedTracker") << "Old z convention: dz --> -dz";
	}
	else
	{
		edm::LogWarning("MisalignedTracker") << "New z convention: dz --> dz";
	}

}

void TrackerSystematicMisalignments::beginJob()
{

}


void TrackerSystematicMisalignments::analyze(const edm::Event& event, const edm::EventSetup& setup){

	//Retrieve tracker topology from geometry
	edm::ESHandle<TrackerTopology> tTopoHandle;
	setup.get<TrackerTopologyRcd>().get(tTopoHandle);
	const TrackerTopology* const tTopo = tTopoHandle.product();

	edm::ESHandle<GeometricDet> geom;
	setup.get<IdealGeometryRecord>().get(geom);
	edm::ESHandle<PTrackerParameters> ptp;
	setup.get<PTrackerParametersRcd>().get( ptp );
	TrackerGeometry* tracker = TrackerGeomBuilderFromGeometricDet().build(&*geom, *ptp, tTopo );

	//take geometry from DB or randomly generate geometry
	if (m_fromDBGeom){
		//build the tracker
		edm::ESHandle<Alignments> alignments;
		edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;

		setup.get<TrackerAlignmentRcd>().get(alignments);
		setup.get<TrackerAlignmentErrorExtendedRcd>().get(alignmentErrors);

		edm::ESHandle<Alignments> globalPositionRcd;
		setup.get<TrackerDigiGeometryRecord>().getRecord<GlobalPositionRcd>().get(globalPositionRcd);

		//apply the latest alignments
		GeometryAligner aligner;
		aligner.applyAlignments<TrackerGeometry>( &(*tracker), &(*alignments), &(*alignmentErrors),
							  align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));

	}

	theAlignableTracker = new AlignableTracker(&(*tracker), tTopo);

	applySystematicMisalignment( &(*theAlignableTracker) );

	// -------------- writing out to alignment record --------------
	Alignments* myAlignments = theAlignableTracker->alignments() ;
	AlignmentErrorsExtended* myAlignmentErrorsExtended = theAlignableTracker->alignmentErrors() ;

	// Store alignment[Error]s to DB
	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	std::string theAlignRecordName = "TrackerAlignmentRcd";
	std::string theErrorRecordName = "TrackerAlignmentErrorExtendedRcd";

	// Call service
	if( !poolDbService.isAvailable() ) // Die if not available
		throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

	poolDbService->writeOne<Alignments>(&(*myAlignments), poolDbService->beginOfTime(), theAlignRecordName);
	poolDbService->writeOne<AlignmentErrorsExtended>(&(*myAlignmentErrorsExtended), poolDbService->beginOfTime(), theErrorRecordName);
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

	// if suppression of blind mvmts: check if subdet is blind to a certain mode
	bool blindToZ(false), blindToR(false);
	if (suppressBlindMvmts)
	{
		const int subdetid = ali->geomDetId().subdetId();
		switch(subdetid)
		{
			// TIB/TON blind to z
			case SiStripDetId::TIB:
			case SiStripDetId::TOB:
				blindToZ = true;
				break;
			// TID/TEC blind to R
			case SiStripDetId::TID:
			case SiStripDetId::TEC:
				blindToR = true;
				break;
			default:
				break;
		}
	}

	const int level = ali->alignableObjectId();
	if ((level == 1)||(level == 2)){
		const align::PositionType gP = ali->globalPosition();
		const align::GlobalVector gVec = findSystematicMis( gP, blindToZ, blindToR);
		ali->move( gVec );
	}
}

align::GlobalVector TrackerSystematicMisalignments::findSystematicMis( const align::PositionType& globalPos, const bool blindToZ, const bool blindToR ){
//align::GlobalVector TrackerSystematicMisalignments::findSystematicMis( align::PositionType globalPos ){
	// calculates shift for the current alignable
	// all corrections are calculated w.r.t. the original geometry
	double deltaX = 0.0;
	double deltaY = 0.0;
	double deltaZ = 0.0;
	const double oldX = globalPos.x();
	const double oldY = globalPos.y();
	const double oldZ = globalPos.z();
	const double oldPhi = globalPos.phi();
	const double oldR = sqrt(globalPos.x()*globalPos.x() + globalPos.y()*globalPos.y());

	if (m_radialEpsilon > -990.0 && !blindToR){
		deltaX += m_radialEpsilon*oldX;
		deltaY += m_radialEpsilon*oldY;
	}
	if (m_telescopeEpsilon > -990.0 && !blindToZ){
		deltaZ += m_telescopeEpsilon*oldR;
	}
	if (m_layerRotEpsilon > -990.0){
		// The following number was chosen such that the Layer Rotation systematic
		// misalignment would not cause an overall rotation of the tracker.
		const double Roffset = 57.0;
		const double xP = oldR*cos(oldPhi+m_layerRotEpsilon*(oldR-Roffset));
		const double yP = oldR*sin(oldPhi+m_layerRotEpsilon*(oldR-Roffset));
		deltaX += (xP - oldX);
		deltaY += (yP - oldY);
	}
	if (m_bowingEpsilon > -990.0 && !blindToR){
		const double trackeredgePlusZ=271.846;
		const double bowfactor=m_bowingEpsilon*(trackeredgePlusZ*trackeredgePlusZ-oldZ*oldZ);
		deltaX += oldX*bowfactor;
		deltaY += oldY*bowfactor;
	}
	if (m_zExpEpsilon > -990.0 && !blindToZ){
		deltaZ += oldZ*m_zExpEpsilon;
	}
	if (m_twistEpsilon > -990.0){
		const double xP = oldR*cos(oldPhi+m_twistEpsilon*oldZ);
		const double yP = oldR*sin(oldPhi+m_twistEpsilon*oldZ);
		deltaX += (xP - oldX);
		deltaY += (yP - oldY);
	}
	if (m_ellipticalEpsilon > -990.0 && !blindToR){
		deltaX += oldX*m_ellipticalEpsilon*cos(2.0*oldPhi + m_ellipticalDelta);
		deltaY += oldY*m_ellipticalEpsilon*cos(2.0*oldPhi + m_ellipticalDelta);
	}
	if (m_skewEpsilon > -990.0 && !blindToZ){
		deltaZ += m_skewEpsilon*cos(oldPhi + m_skewDelta);
	}
	if (m_sagittaEpsilon > -990.0){
		// deltaX += oldX/fabs(oldX)*m_sagittaEpsilon; // old one...
		deltaX += oldR*m_sagittaEpsilon*sin(m_sagittaDelta);
		deltaY += oldR*m_sagittaEpsilon*cos(m_sagittaDelta);    //Delta y is cos so that delta=0 reflects the old behavior
	}

	// Compatibility with old version <= 1.5
	if (oldMinusZconvention) deltaZ = -deltaZ;

	align::GlobalVector gV( deltaX, deltaY, deltaZ );
	return gV;
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackerSystematicMisalignments);
