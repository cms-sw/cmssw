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
 */

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"

namespace edm {class ParameterSet; class EventSetup;}

class MuonServiceProxy {
public:
  /// Constructor
  MuonServiceProxy(const edm::ParameterSet& par);

  /// Destructor
  virtual ~MuonServiceProxy();

  // Operations
  
  /// update the services each event
  void update(const edm::EventSetup& setup);

  /// get the magnetic field
  edm::ESHandle<MagneticField> magneticField() const {return theMGField;}

  /// get the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const {return theTrackingGeometry;}

  /// get the detLayer geometry
  edm::ESHandle<MuonDetLayerGeometry> detLayerGeometry() const {return theDetLayerGeometry;}

  /// get the propagator
  edm::ESHandle<Propagator> propagator(std::string propagatorName) const;

  /// get the whole EventSetup
  const edm::EventSetup &eventSetup() const {return *theEventSetup;}
  
  /// check if the MuonReco Geometry has been changed
  bool isTrackingComponentsRecordChanged() const {return theChangeInTrackingComponentsRecord;}
  
  const MuonNavigationSchool *muonNavigationSchool() const{
    return theSchool;
  }

 protected:
  
 private:
  typedef std::map<std::string,  edm::ESHandle<Propagator> > propagators;
  
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
  const edm::EventSetup *theEventSetup;
  bool theMuonNavigationFlag;
  bool theRPCLayer;
  bool theCSCLayer;
  bool theGEMLayer;
  const MuonNavigationSchool* theSchool;

  propagators thePropagators;

  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_DG;
  unsigned long long theCacheId_P;
  
  bool theChangeInTrackingComponentsRecord;

};
#endif

