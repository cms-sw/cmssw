#ifndef RecoMuon_TrackingTools_MuonServiceProxy_H
#define RecoMuon_TrackingTools_MuonServiceProxy_H

/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  \author N. Amapane - CERN <nicola.amapane@cern.ch>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *
 *  Modified by C. Calabria
 *  Modified by D. Nash
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// EventSetup data types
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// EventSetup record types
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"

#include <map>
#include <string>

class MuonServiceProxy {
public:
  enum class UseEventSetupIn { Run, Event, RunAndEvent };

  /// Constructor
  MuonServiceProxy(const edm::ParameterSet&,
                   edm::ConsumesCollector&&,
                   UseEventSetupIn useEventSetupIn = UseEventSetupIn::Event);

  /// Destructor
  virtual ~MuonServiceProxy();

  // Operations

  /// update the services each event
  void update(const edm::EventSetup& setup, bool duringEvent = true);

  /// get the magnetic field
  edm::ESHandle<MagneticField> magneticField() const { return theMGField; }

  /// get the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const { return theTrackingGeometry; }

  /// get the detLayer geometry
  edm::ESHandle<MuonDetLayerGeometry> detLayerGeometry() const { return theDetLayerGeometry; }

  /// get the propagator
  edm::ESHandle<Propagator> propagator(std::string propagatorName) const;

  /// get the whole EventSetup
  /// (Note: this is a dangerous function. I would delete it if modules were
  /// not using it. If this function is called for an event where the function
  /// 'update' was not called, then the pointer stored in 'theEventSetup' will point to
  /// an object that no longer exists even if all the ESHandles are still valid!
  /// Be careful. As long as 'update' is called every event and this is only
  /// used while processing that single corresponding event, it will work OK...
  /// This function also makes it difficult to examine code in a module and
  /// understand which parts of a module use the EventSetup to get data.)
  const edm::EventSetup& eventSetup() const { return *theEventSetup; }

  /// check if the MuonReco Geometry has been changed
  bool isTrackingComponentsRecordChanged() const { return theChangeInTrackingComponentsRecord; }

  const MuonNavigationSchool* muonNavigationSchool() const { return theSchool; }

private:
  class PropagatorInfo {
  public:
    edm::ESHandle<Propagator> esHandle_;
    edm::ESGetToken<Propagator, TrackingComponentsRecord> eventToken_;
    edm::ESGetToken<Propagator, TrackingComponentsRecord> runToken_;
  };

  using PropagatorMap = std::map<std::string, PropagatorInfo>;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> globalTrackingGeometryEventToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldEventToken_;
  edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonDetLayerGeometryEventToken_;

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> globalTrackingGeometryRunToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldRunToken_;
  edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonDetLayerGeometryRunToken_;

  const edm::EventSetup* theEventSetup;
  bool theMuonNavigationFlag;
  bool theRPCLayer;
  bool theCSCLayer;
  bool theGEMLayer;
  bool theME0Layer;
  const MuonNavigationSchool* theSchool;

  PropagatorMap thePropagators;

  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_DG;
  unsigned long long theCacheId_P;

  bool theChangeInTrackingComponentsRecord;
};
#endif
