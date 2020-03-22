/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  \author N. Amapane - CERN <nicola.amapane@cern.ch>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *
 *  Modified by C. Calabria & A. Sharma
 *  Modified by D. Nash
 */

// Class Header
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

// Framework Headers
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include <vector>

// Constructor
MuonServiceProxy::MuonServiceProxy(const edm::ParameterSet& par,
                                   edm::ConsumesCollector&& iC,
                                   UseEventSetupIn useEventSetupIn)
    : theTrackingGeometry(nullptr),
      theMGField(nullptr),
      theDetLayerGeometry(nullptr),
      theEventSetup(nullptr),
      theSchool(nullptr) {
  theMuonNavigationFlag = par.getUntrackedParameter<bool>("UseMuonNavigation", true);

  if (theMuonNavigationFlag) {
    theRPCLayer = par.getParameter<bool>("RPCLayers");

    if (par.existsAs<bool>("CSCLayers"))
      theCSCLayer = par.getParameter<bool>("CSCLayers");
    else
      theCSCLayer = true;

    if (par.existsAs<bool>("GEMLayers"))
      theGEMLayer = par.getParameter<bool>("GEMLayers");
    else
      theGEMLayer = false;

    if (par.existsAs<bool>("ME0Layers"))
      theME0Layer = par.getParameter<bool>("ME0Layers");
    else
      theME0Layer = false;

  } else {
    theRPCLayer = true;
    theCSCLayer = true;
    theGEMLayer = true;
    theME0Layer = true;
  }

  // load the propagators map
  std::vector<std::string> noPropagators;
  std::vector<std::string> propagatorNames =
      par.getUntrackedParameter<std::vector<std::string>>("Propagators", noPropagators);

  if (propagatorNames.empty())
    LogDebug("Muon|RecoMuon|MuonServiceProxy") << "NO propagator(s) selected!";

  for (auto const& propagatorName : propagatorNames) {
    thePropagators[propagatorName] = PropagatorInfo();
  }
  theCacheId_GTG = 0;
  theCacheId_MG = 0;
  theCacheId_DG = 0;
  theCacheId_P = 0;
  theChangeInTrackingComponentsRecord = false;

  // Declare the products we get from the EventSetup and initialize the tokens used to get them
  if (useEventSetupIn == UseEventSetupIn::Event || useEventSetupIn == UseEventSetupIn::RunAndEvent) {
    globalTrackingGeometryEventToken_ = iC.esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>();
    magneticFieldEventToken_ = iC.esConsumes<MagneticField, IdealMagneticFieldRecord>();
    muonDetLayerGeometryEventToken_ = iC.esConsumes<MuonDetLayerGeometry, MuonRecoGeometryRecord>();
    for (auto& element : thePropagators) {
      element.second.eventToken_ =
          iC.esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", element.first));
    }
  }
  if (useEventSetupIn == UseEventSetupIn::Run || useEventSetupIn == UseEventSetupIn::RunAndEvent) {
    globalTrackingGeometryRunToken_ =
        iC.esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord, edm::Transition::BeginRun>();
    magneticFieldRunToken_ = iC.esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>();
    muonDetLayerGeometryRunToken_ =
        iC.esConsumes<MuonDetLayerGeometry, MuonRecoGeometryRecord, edm::Transition::BeginRun>();
    for (auto& element : thePropagators) {
      element.second.runToken_ = iC.esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", element.first));
    }
  }
}

// Destructor
MuonServiceProxy::~MuonServiceProxy() {
  if (theSchool)
    delete theSchool;
}

// Operations

// update the services each event
void MuonServiceProxy::update(const edm::EventSetup& setup, bool duringEvent) {
  const std::string metname = "Muon|RecoMuon|MuonServiceProxy";

  theEventSetup = &setup;

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if (newCacheId_GTG != theCacheId_GTG) {
    LogTrace(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    if (duringEvent) {
      theTrackingGeometry = setup.getHandle(globalTrackingGeometryEventToken_);
    } else {
      theTrackingGeometry = setup.getHandle(globalTrackingGeometryRunToken_);
    }
  }

  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if (newCacheId_MG != theCacheId_MG) {
    LogTrace(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    if (duringEvent) {
      theMGField = setup.getHandle(magneticFieldEventToken_);
    } else {
      theMGField = setup.getHandle(magneticFieldRunToken_);
    }
  }

  // DetLayer Geometry
  unsigned long long newCacheId_DG = setup.get<MuonRecoGeometryRecord>().cacheIdentifier();
  if (newCacheId_DG != theCacheId_DG) {
    LogTrace(metname) << "Muon Reco Geometry changed!";
    theCacheId_DG = newCacheId_DG;
    if (duringEvent) {
      theDetLayerGeometry = setup.getHandle(muonDetLayerGeometryEventToken_);
    } else {
      theDetLayerGeometry = setup.getHandle(muonDetLayerGeometryRunToken_);
    }
    // MuonNavigationSchool should live until its validity expires, and then DELETE
    // the NavigableLayers (this is implemented in MuonNavigationSchool's dtor)
    if (theMuonNavigationFlag) {
      if (theSchool)
        delete theSchool;
      theSchool = new MuonNavigationSchool(&*theDetLayerGeometry, theRPCLayer, theCSCLayer, theGEMLayer, theME0Layer);
    }
  }

  // Propagators
  unsigned long long newCacheId_P = setup.get<TrackingComponentsRecord>().cacheIdentifier();
  if (newCacheId_P != theCacheId_P) {
    LogTrace(metname) << "Tracking Component changed!";
    theChangeInTrackingComponentsRecord = true;
    theCacheId_P = newCacheId_P;
    for (auto& element : thePropagators) {
      if (duringEvent) {
        element.second.esHandle_ = setup.getHandle(element.second.eventToken_);
      } else {
        element.second.esHandle_ = setup.getHandle(element.second.runToken_);
      }
    }
  } else
    theChangeInTrackingComponentsRecord = false;
}

// get the propagator
edm::ESHandle<Propagator> MuonServiceProxy::propagator(std::string propagatorName) const {
  PropagatorMap::const_iterator prop = thePropagators.find(propagatorName);

  if (prop == thePropagators.end()) {
    edm::LogError("Muon|RecoMuon|MuonServiceProxy") << "MuonServiceProxy: propagator with name: " << propagatorName
                                                    << " not found! Please load it in the MuonServiceProxy.cff";
    return edm::ESHandle<Propagator>(nullptr);
  }
  return prop->second.esHandle_;
}
