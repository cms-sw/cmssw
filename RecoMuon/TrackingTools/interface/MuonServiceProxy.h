#ifndef RecoMuon_TrackingTools_MuonServiceProxy_H
#define RecoMuon_TrackingTools_MuonServiceProxy_H

/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN <nicola.amapane@cern.ch>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

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
  edm::ESHandle<MagneticField> magneticField() {return theMGField;}

  /// get the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() {return theTrackingGeometry;}

  /// get the detLayer geometry
  edm::ESHandle<MuonDetLayerGeometry> detLayerGeometry() {return theDetLayerGeometry;}

  /// get the propagator
  edm::ESHandle<Propagator> propagator(std::string propagatorName);

protected:

private:
  typedef std::map<std::string,  edm::ESHandle<Propagator> > propagators;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
  propagators thePropagators;

};
#endif

