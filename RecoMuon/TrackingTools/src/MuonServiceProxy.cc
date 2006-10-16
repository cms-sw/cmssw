/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  $Date: 2006/10/13 15:00:05 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN <nicola.amapane@cern.ch>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Class Header
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

// Service Records
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

// Framework Headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// C++ Headers
#include <map>

using namespace std;
using namespace edm;

// Constructor
MuonServiceProxy::MuonServiceProxy(const edm::ParameterSet& par):theTrackingGeometry(0),theMGField(0),theDetLayerGeometry(0),theEventSetup(0){

  // load the propagators map
  vector<string> noPropagators;
  vector<string> propagatorNames;

  propagatorNames = par.getUntrackedParameter<vector<string> >("Propagators", noPropagators);
  
  if(propagatorNames.empty())
    LogError("Muon|RecoMuon|MuonServiceProxy") << "NO propagator(s) selected!";
  
  for(vector<string>::iterator propagatorName = propagatorNames.begin();
      propagatorName != propagatorNames.end(); ++propagatorName)
    thePropagators[ *propagatorName ] = ESHandle<Propagator>(0);

  theCacheId_GTG = 0;
  theCacheId_MG = 0;  
  theCacheId_DG = 0;
  theCacheId_P = 0;
}


// Destructor
MuonServiceProxy::~MuonServiceProxy(){
  
  // FIXME: how do that?
  // delete theTrackingGeometry;
  // delete theMGField;
  // delete theDetLayerGeometry;
  
  // FIXME: is it enough?
  thePropagators.clear();
}

// Operations

// update the services each event
void MuonServiceProxy::update(const edm::EventSetup& setup){
  const std::string metname = "Muon|RecoMuon|MuonServiceProxy";
  
  theEventSetup = &setup;

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != theCacheId_GTG ) {
    LogDebug(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  
  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if ( newCacheId_MG != theCacheId_MG ) {
    LogDebug(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // DetLayer Geometry
  unsigned long long newCacheId_DG = setup.get<MuonRecoGeometryRecord>().cacheIdentifier();
  if ( newCacheId_DG != theCacheId_DG ) {
    LogDebug(metname) << "Muon Reco Geometry changed!";
    theCacheId_DG = newCacheId_DG;
    setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
    // FIXME: MuonNavigationSchool should live until its validity expires, and then DELETE
    // the NavigableLayers (this is not implemented in MuonNavigationSchool's dtor)
    // i.e. should become a pointer member here
    // the setter should be called at each event, if there is more than one navigation type!!
    MuonNavigationSchool school(&*theDetLayerGeometry);
    NavigationSetter setter(school); 
  }
  
  // Propagators
  unsigned long long newCacheId_P = setup.get<TrackingComponentsRecord>().cacheIdentifier();
  if ( newCacheId_P != theCacheId_P ) {
    LogDebug(metname) << "Tracking Component changed!";
    theCacheId_P = newCacheId_P;
    for(propagators::iterator prop = thePropagators.begin(); prop != thePropagators.end();
	++prop)
      setup.get<TrackingComponentsRecord>().get( prop->first , prop->second );
  }
}

// get the propagator
ESHandle<Propagator> MuonServiceProxy::propagator(std::string propagatorName) const{
  
  propagators::const_iterator prop = thePropagators.find(propagatorName);

  if (prop == thePropagators.end()) return 0;
  
  return prop->second;
}


