/** \file
 *
 *  $Date: 2006/06/26 13:24:25 $
 *  $Revision: 1.2 $
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"


#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "RecoMuon/DetLayers/interface/MuRingForwardLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"

#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

//#include "Geometry/Vector/interface/CoordinateSets.h"


#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace edm;

class MuonRecoGeometryAnalyzer : public EDAnalyzer {
 public:

  MuonRecoGeometryAnalyzer( const ParameterSet& pset);

  virtual void analyze( const Event& ev, const EventSetup& es);

  void testDTLayers(const MuonDetLayerGeometry*, const MagneticField* field);
  void testCSCLayers(const MuonDetLayerGeometry*, const MagneticField* field);

  MeasurementEstimator *theEstimator;
};


  
MuonRecoGeometryAnalyzer::MuonRecoGeometryAnalyzer(const ParameterSet& iConfig) 
{
  float theMaxChi2=25.;
  float theNSigma=3.;
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2,theNSigma);
  
}


void MuonRecoGeometryAnalyzer::analyze( const Event& ev,
				       const EventSetup& es ) {

  ESHandle<MuonDetLayerGeometry> geo;
  es.get<MuonRecoGeometryRecord>().get(geo);

  ESHandle<MagneticField> magfield;
  es.get<IdealMagneticFieldRecord>().get(magfield);
  
  //  testDTLayers(geo.product(),magfield.product());
  testCSCLayers(geo.product(),magfield.product());
}


void MuonRecoGeometryAnalyzer::testDTLayers(const MuonDetLayerGeometry* geo,const MagneticField* field) {

  const vector<DetLayer*>& layers = geo->allDTLayers();

  for (vector<DetLayer*>::const_iterator ilay = layers.begin(); ilay!=layers.end(); ++ilay) {
    const MuRodBarrelLayer* layer = (const MuRodBarrelLayer*) (*ilay);
  
    const BoundCylinder& cyl = layer->specificSurface();  

    double halfZ = cyl.bounds().length()/2.;

    // Generate a random point on the cylinder
    double aPhi = RandFlat::shoot(-Geom::pi(),Geom::pi());
    double aZ = RandFlat::shoot(-halfZ, halfZ);
    GlobalPoint gp(GlobalPoint::Cylindrical(cyl.radius(), aPhi, aZ));  

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp,gv,charge,field);
    TrajectoryStateOnSurface tsos(gtp, cyl);
    cout << "testDTLayers: at " << tsos.globalPosition()
	 << " R=" << tsos.globalPosition().perp()
	 << " phi=" << tsos.globalPosition().phi()
	 << " Z=" << tsos.globalPosition().z()
	 << " p = " << tsos.globalMomentum()
	 << endl;


    SteppingHelixPropagator prop(field,alongMomentum);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos,prop,*theEstimator);
    cout << "is compatible: " << comp.first
	 << " at: R=" << comp.second.globalPosition().perp()
	 << " phi=" << comp.second.globalPosition().phi()
	 << " Z=" <<  comp.second.globalPosition().z()
	 << endl;
  
    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos,prop,*theEstimator);
    if (compDets.size()) {
      cout << "compatibleDets: " << compDets.size() << endl

	   << "  final state pos: " << compDets.front().second.globalPosition() << endl 
	   << "  det         pos: " << compDets.front().first->position()
	   << " id: " << DTWireId(compDets.front().first->geographicalId().rawId()) << endl 
	   << "  distance " << (tsos.globalPosition()-compDets.front().first->position()).mag()

	   << endl
	   << endl;
    } else {
      cout << " ERROR : no compatible det found" << endl;
    }    
  }
}

void MuonRecoGeometryAnalyzer::testCSCLayers(const MuonDetLayerGeometry* geo,const MagneticField* field) {

  const vector<DetLayer*>& layers = geo->allCSCLayers();

  for (vector<DetLayer*>::const_iterator ilay = layers.begin(); ilay!=layers.end(); ++ilay) {
    const MuRingForwardLayer* layer = (const MuRingForwardLayer*) (*ilay);
  
    const BoundDisk& disk = layer->specificSurface();

    // Generate a random point on the disk
    double aPhi = RandFlat::shoot(-Geom::pi(),Geom::pi());
    double aR = RandFlat::shoot(disk.innerRadius(), disk.outerRadius());
    GlobalPoint gp(GlobalPoint::Cylindrical(aR, aPhi, disk.position().z()));  

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp,gv,charge,field);
    TrajectoryStateOnSurface tsos(gtp, disk);
    cout << "testCSCLayers: at " << tsos.globalPosition()
	 << " R=" << tsos.globalPosition().perp()
	 << " phi=" << tsos.globalPosition().phi()
	 << " Z=" << tsos.globalPosition().z()
	 << " p = " << tsos.globalMomentum()
	 << endl;


    SteppingHelixPropagator prop(field,alongMomentum);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos,prop,*theEstimator);
    cout << "is compatible: " << comp.first
	 << " at: R=" << comp.second.globalPosition().perp()
	 << " phi=" << comp.second.globalPosition().phi()
	 << " Z=" <<  comp.second.globalPosition().z()
	 << endl;
  
    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos,prop,*theEstimator);
    if (compDets.size()) {
      cout << "compatibleDets: " << compDets.size() << endl

	   << "  final state pos: " << compDets.front().second.globalPosition() << endl 
	   << "  det         pos: " << compDets.front().first->position()
	   << " id: " << CSCDetId(compDets.front().first->geographicalId().rawId()) << endl 
	   << "  distance " << (tsos.globalPosition()-compDets.front().first->position()).mag()

	   << endl
	   << endl;
    } else {
      cout << " ERROR : no compatible det found" << endl;
    }
  }
}




//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MuonRecoGeometryAnalyzer)
