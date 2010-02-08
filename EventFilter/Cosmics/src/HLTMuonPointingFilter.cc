/** \file
 *
 * $Date:  07/11/2007 15:14:20 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "EventFilter/Cosmics/interface/HLTMuonPointingFilter.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
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
HLTMuonPointingFilter::HLTMuonPointingFilter(const edm::ParameterSet& pset) {

  // the name of the STA rec hits collection
  theSTAMuonLabel = pset.getParameter<string>("SALabel");

  thePropagatorName = pset.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  theRadius = pset.getParameter<double>("radius"); // cyl's radius (cm)
  theMaxZ = pset.getParameter<double>("maxZ"); // cyl's half lenght (cm)

  LogDebug("HLTMuonPointing") << " SALabel : " << theSTAMuonLabel 
    << " Radius : " << theRadius
    << " Half lenght : " << theMaxZ;
}

/// Destructor
HLTMuonPointingFilter::~HLTMuonPointingFilter() {
}

/* Operations */ 
bool HLTMuonPointingFilter::filter(edm::Event& event, const edm::EventSetup& eventSetup) {
  bool accept = false;
  if (!thePropagator){
    ESHandle<Propagator> prop;
    eventSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
    thePropagator = prop->clone();
    thePropagator->setPropagationDirection(anyDirection);
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

    // Get a surface (here a cylinder of radius 1290mm) ECAL
    Cylinder::PositionType pos0;
    Cylinder::RotationType rot0;
    const Cylinder::CylinderPointer cyl = Cylinder::build(pos0, rot0, theRadius);

    TrajectoryStateOnSurface tsosAtCyl =
      thePropagator->propagate(*innerTSOS.freeState(), *cyl);

    if ( tsosAtCyl.isValid() ) {
      LogDebug("HLTMuonPointing") << " extrap TSOS " << tsosAtCyl;
      if (fabs(tsosAtCyl.globalPosition().z())<theMaxZ ) {
        accept=true;
        return accept;
      }
      else { 
        LogDebug("HLTMuonPointing") << " extrap TSOS z too big " << tsosAtCyl.globalPosition().z();
      }
    } else {
      LogDebug("HLTMuonPointing") << " extrap to cyl failed ";
    }

  }

  return accept;


}

// define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonPointingFilter);
