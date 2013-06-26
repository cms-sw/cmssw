/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  $Date: 2009/10/14 10:34:51 $
 *  $Revision: 1.19 $
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
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

// Framework Headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

// C++ Headers
#include <map>

using namespace std;
using namespace edm;

// Constructor
MuonServiceProxy::MuonServiceProxy(const edm::ParameterSet& par):theTrackingGeometry(0),theMGField(0),theDetLayerGeometry(0),theEventSetup(0),theSchool(0){
  
  // load the propagators map
  vector<string> noPropagators;
  vector<string> propagatorNames;

  theMuonNavigationFlag = par.getUntrackedParameter<bool>("UseMuonNavigation",true);
  if(theMuonNavigationFlag) theRPCLayer = par.getParameter<bool>("RPCLayers");
  else theRPCLayer = true;

  propagatorNames = par.getUntrackedParameter<vector<string> >("Propagators", noPropagators);
  
  if(propagatorNames.empty())
    LogDebug("Muon|RecoMuon|MuonServiceProxy") << "NO propagator(s) selected!";
  
  for(vector<string>::iterator propagatorName = propagatorNames.begin();
      propagatorName != propagatorNames.end(); ++propagatorName)
    thePropagators[ *propagatorName ] = ESHandle<Propagator>(0);

  theCacheId_GTG = 0;
  theCacheId_MG = 0;  
  theCacheId_DG = 0;
  theCacheId_P = 0;
  theChangeInTrackingComponentsRecord = false;

}


// Destructor
MuonServiceProxy::~MuonServiceProxy(){
  
  // FIXME: how do that?
  // delete theTrackingGeometry;
  // delete theMGField;
  // delete theDetLayerGeometry;
  
  // FIXME: is it enough?
  thePropagators.clear();
  if(theSchool) delete theSchool;
}

// Operations

// update the services each event
void MuonServiceProxy::update(const edm::EventSetup& setup){
  const std::string metname = "Muon|RecoMuon|MuonServiceProxy";
  
  theEventSetup = &setup;

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != theCacheId_GTG ) {
    LogTrace(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  
  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if ( newCacheId_MG != theCacheId_MG ) {
    LogTrace(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // DetLayer Geometry
  unsigned long long newCacheId_DG = setup.get<MuonRecoGeometryRecord>().cacheIdentifier();
  if ( newCacheId_DG != theCacheId_DG ) {
    LogTrace(metname) << "Muon Reco Geometry changed!";
    theCacheId_DG = newCacheId_DG;
    setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
    // MuonNavigationSchool should live until its validity expires, and then DELETE
    // the NavigableLayers (this is implemented in MuonNavigationSchool's dtor)
    if ( theMuonNavigationFlag ) {
      if(theSchool) delete theSchool;
      theSchool = new MuonNavigationSchool(&*theDetLayerGeometry,theRPCLayer);
    }
  }
  
  // Propagators
  unsigned long long newCacheId_P = setup.get<TrackingComponentsRecord>().cacheIdentifier();
  if ( newCacheId_P != theCacheId_P ) {
    LogTrace(metname) << "Tracking Component changed!";
    theChangeInTrackingComponentsRecord = true;
    theCacheId_P = newCacheId_P;
    for(propagators::iterator prop = thePropagators.begin(); prop != thePropagators.end();
	++prop)
      setup.get<TrackingComponentsRecord>().get( prop->first , prop->second );
  }
  else
    theChangeInTrackingComponentsRecord = false;

}

// get the propagator
ESHandle<Propagator> MuonServiceProxy::propagator(std::string propagatorName) const{
  
  propagators::const_iterator prop = thePropagators.find(propagatorName);
  
  if (prop == thePropagators.end()){
    LogError("Muon|RecoMuon|MuonServiceProxy") 
      << "MuonServiceProxy: propagator with name: "<< propagatorName <<" not found! Please load it in the MuonServiceProxy.cff"; 
    return ESHandle<Propagator>(0);
  }
  
  return prop->second;
}


 
