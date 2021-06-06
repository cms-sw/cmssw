/** \file
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "HLTMuonPointingFilter.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

/* C++ Headers */
using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
HLTMuonPointingFilter::HLTMuonPointingFilter(const edm::ParameterSet& pset)
    : theSTAMuonToken(
          consumes<reco::TrackCollection>(pset.getParameter<edm::InputTag>("SALabel"))),  // token to read the muons
      thePropagatorName(pset.getParameter<std::string>("PropagatorName")),
      thePropagatorToken(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", thePropagatorName))),
      theMGFieldToken(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      theTrackingGeometryToken(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()),
      theRadius(pset.getParameter<double>("radius")),            // cyl's radius (cm)
      theMaxZ(pset.getParameter<double>("maxZ")),                // cyl's half lenght (cm)
      thePixHits(pset.getParameter<unsigned int>("PixHits")),    // pixel hits
      theTkLayers(pset.getParameter<unsigned int>("TkLayers")),  // tracker layers with measurements
      theMuonHits(pset.getParameter<unsigned int>("MuonHits")),  // muon hits
      // Get a surface (here a cylinder of radius 1290mm) ECAL
      theCyl(Cylinder::build(theRadius, Cylinder::PositionType{}, Cylinder::RotationType{})),
      thePosPlane(Plane::build(Plane::PositionType{0, 0, static_cast<float>(theMaxZ)}, Plane::RotationType{})),
      theNegPlane(Plane::build(Plane::PositionType{0, 0, static_cast<float>(-theMaxZ)}, Plane::RotationType{})) {
  LogDebug("HLTMuonPointing") << " SALabel : " << pset.getParameter<edm::InputTag>("SALabel")
                              << " Radius : " << theRadius << " Half lenght : " << theMaxZ
                              << " Min pixel hits : " << thePixHits << " Min tk layers measurements : " << theTkLayers
                              << " Min muon hits : " << theMuonHits;
}

/// Destructor
HLTMuonPointingFilter::~HLTMuonPointingFilter() = default;

/* Operations */
bool HLTMuonPointingFilter::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& eventSetup) const {
  bool accept = false;

  auto const& propagator = eventSetup.getData(thePropagatorToken);

  if (propagator.propagationDirection() != anyDirection) {
    throw cms::Exception("Configuration")
        << "the propagator " << thePropagatorName
        << " should be configured with PropagationDirection = \"anyDirection\"" << std::endl;
  }

  ESHandle<MagneticField> theMGField = eventSetup.getHandle(theMGFieldToken);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry = eventSetup.getHandle(theTrackingGeometryToken);

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByToken(theSTAMuonToken, staTracks);

  reco::TrackCollection::const_iterator staTrack;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
    reco::TransientTrack track(*staTrack, &*theMGField, theTrackingGeometry);

    const reco::HitPattern& p = track.hitPattern();

    const unsigned int pixelHits = p.numberOfValidPixelHits();
    const unsigned int trkLayers = p.trackerLayersWithMeasurement();
    const unsigned int nMuonHits = p.numberOfValidMuonHits();

    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();

    LogDebug("HLTMuonPointing") << " InnerTSOS " << innerTSOS;

    TrajectoryStateOnSurface tsosAtCyl = propagator.propagate(*innerTSOS.freeState(), *theCyl);

    if (tsosAtCyl.isValid()) {
      LogDebug("HLTMuonPointing") << " extrap TSOS " << tsosAtCyl << " number of pixel hits " << pixelHits
                                  << " number of tracker layers with interactions " << trkLayers
                                  << " number of muon hits " << nMuonHits;
      if (fabs(tsosAtCyl.globalPosition().z()) < theMaxZ) {
        if (pixelHits >= thePixHits) {
          if (trkLayers >= theTkLayers) {
            if (nMuonHits >= theMuonHits) {
              accept = true;
              return accept;
            }
          }
        }
      } else {
        LogDebug("HLTMuonPointing") << " extrap TSOS z too big " << tsosAtCyl.globalPosition().z()
                                    << " number of pixel hits " << pixelHits
                                    << " number of tracker layers with interactions " << trkLayers
                                    << " number of muon hits " << nMuonHits;
        TrajectoryStateOnSurface tsosAtPlane;
        if (tsosAtCyl.globalPosition().z() > 0)
          tsosAtPlane = propagator.propagate(*innerTSOS.freeState(), *thePosPlane);
        else
          tsosAtPlane = propagator.propagate(*innerTSOS.freeState(), *theNegPlane);

        if (tsosAtPlane.isValid()) {
          if (tsosAtPlane.globalPosition().perp() < theRadius) {
            if (pixelHits >= thePixHits) {
              if (trkLayers >= theTkLayers) {
                if (nMuonHits >= theMuonHits) {
                  accept = true;
                  return accept;
                }
              }
            }
          }
        } else
          LogDebug("HLTMuonPointing") << " extrap to plane failed ";
      }
    } else {
      LogDebug("HLTMuonPointing") << " extrap to cyl failed ";
    }
  }

  return accept;
}

void HLTMuonPointingFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("SALabel", edm::InputTag("hltCosmicMuonBarrelOnly"));
  desc.add<std::string>("PropagatorName", "SteppingHelixPropagatorAny");
  desc.add<double>("radius", 90.0);
  desc.add<double>("maxZ", 280.0);
  desc.add<unsigned int>("PixHits", 0);
  desc.add<unsigned int>("TkLayers", 0);
  desc.add<unsigned int>("MuonHits", 0);

  descriptions.add("hltMuonPointingFilter", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonPointingFilter);
