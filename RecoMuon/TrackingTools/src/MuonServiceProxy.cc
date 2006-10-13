/** \class MuonServiceProxy
 *  Class to handle the services needed by the muon reconstruction
 *  This class avoid the EventSetup percolation.
 *  The update method is called each event in order to update the
 *  pointers.
 *
 *  $Date: 2006/09/01 16:24:12 $
 *  $Revision: 1.4 $
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
  theEventSetup = &setup;

  // Global Tracking Geometry
  static unsigned long long oldCacheId_GTG = 0;
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != oldCacheId_GTG ) {
    oldCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  

  // Magfield
  {
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // DetLayer Geometry
  {
    setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
    // FIXME: MuonNavigationSchool should live until its validity expires, and then DELETE
    // the NavigableLayers (this is not implemented in MuonNavigationSchool's dtor)
    // i.e. should become a pointer member here
    // the setter should be called at each event, if there is more than one navigation type!!
    MuonNavigationSchool school(&*theDetLayerGeometry);
    NavigationSetter setter(school); 
  }
  
  for(propagators::iterator prop = thePropagators.begin(); prop != thePropagators.end();
      ++prop)
    setup.get<TrackingComponentsRecord>().get( prop->first , prop->second );
}

// get the propagator
ESHandle<Propagator> MuonServiceProxy::propagator(std::string propagatorName) const{
  
  propagators::const_iterator prop = thePropagators.find(propagatorName);

  if (prop == thePropagators.end()) return 0;
  
  return prop->second;
}


