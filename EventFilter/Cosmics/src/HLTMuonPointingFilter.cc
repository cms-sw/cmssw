/** \file
 *
 * $Date: 2012/12/26 14:10:36 $
 * $Revision: 1.6 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "EventFilter/Cosmics/interface/HLTMuonPointingFilter.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"



/* C++ Headers */
using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
HLTMuonPointingFilter::HLTMuonPointingFilter(const edm::ParameterSet& pset) :
   HLTFilter(pset),
   m_cacheRecordId(0) 
{
  // the name of the STA rec hits collection
  theSTAMuonLabel = pset.getParameter<string>("SALabel");

  thePropagatorName = pset.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  theRadius = pset.getParameter<double>("radius"); // cyl's radius (cm)
  theMaxZ = pset.getParameter<double>("maxZ"); // cyl's half lenght (cm)


  // Get a surface (here a cylinder of radius 1290mm) ECAL
  Cylinder::PositionType pos0;
  Cylinder::RotationType rot0;
  theCyl = Cylinder::build(theRadius, pos0, rot0);
    
  Plane::PositionType posPos(0,0,theMaxZ);
  Plane::PositionType posNeg(0,0,-theMaxZ);

  thePosPlane = Plane::build(posPos,rot0);
  theNegPlane = Plane::build(posNeg,rot0);

  LogDebug("HLTMuonPointing") << " SALabel : " << theSTAMuonLabel 
    << " Radius : " << theRadius
    << " Half lenght : " << theMaxZ;
}

/// Destructor
HLTMuonPointingFilter::~HLTMuonPointingFilter() {
}

/* Operations */ 
bool HLTMuonPointingFilter::hltFilter(edm::Event& event, const edm::EventSetup& eventSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {
  bool accept = false;

  const TrackingComponentsRecord & tkRec = eventSetup.get<TrackingComponentsRecord>();
  if (not thePropagator or tkRec.cacheIdentifier() != m_cacheRecordId) {
    ESHandle<Propagator> prop;
    tkRec.get(thePropagatorName, prop);
    thePropagator = prop->clone();
    thePropagator->setPropagationDirection(anyDirection);
    m_cacheRecordId = tkRec.cacheIdentifier();
  }

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);

  reco::TrackCollection::const_iterator staTrack;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry);

    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();

    LogDebug("HLTMuonPointing") << " InnerTSOS " << innerTSOS;

    TrajectoryStateOnSurface tsosAtCyl =
      thePropagator->propagate(*innerTSOS.freeState(), *theCyl);

    if ( tsosAtCyl.isValid() ) {
      LogDebug("HLTMuonPointing") << " extrap TSOS " << tsosAtCyl;
      if (fabs(tsosAtCyl.globalPosition().z())<theMaxZ ) {
        accept=true;
        return accept;
      }
      else { 
        LogDebug("HLTMuonPointing") << " extrap TSOS z too big " << tsosAtCyl.globalPosition().z();
	TrajectoryStateOnSurface tsosAtPlane;
	if (tsosAtCyl.globalPosition().z()>0)
	  tsosAtPlane=thePropagator->propagate(*innerTSOS.freeState(), *thePosPlane);
	else
	  tsosAtPlane=thePropagator->propagate(*innerTSOS.freeState(), *theNegPlane);

	if (tsosAtPlane.isValid()){
	  if (tsosAtPlane.globalPosition().perp()< theRadius){
	    accept=true;
	    return accept;
	  }
	}
	else
	  LogDebug("HLTMuonPointing") << " extrap to plane failed ";
      }
    } else {
      LogDebug("HLTMuonPointing") << " extrap to cyl failed ";
    }

  }

  return accept;


}

// define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonPointingFilter);
