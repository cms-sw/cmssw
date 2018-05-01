/** \file
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"


#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetRing.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/ForwardDetId/interface/BTLDetId.h>
#include <DataFormats/ForwardDetId/interface/ETLDetId.h>

#include <sstream>

#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace edm;

class MTDRecoGeometryAnalyzer : public EDAnalyzer {
 public:

  MTDRecoGeometryAnalyzer( const ParameterSet& pset);

  virtual void analyze( const Event& ev, const EventSetup& es);

  void testBTLLayers(const MTDDetLayerGeometry*, const MagneticField* field);
  void testETLLayers(const MTDDetLayerGeometry*, const MagneticField* field);

  string dumpLayer(const DetLayer* layer) const;

 private:
  MeasurementEstimator *theEstimator;
};


  
MTDRecoGeometryAnalyzer::MTDRecoGeometryAnalyzer(const ParameterSet& iConfig) 
{
  float theMaxChi2=25.;
  float theNSigma=3.;
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2,theNSigma);
  
}


void MTDRecoGeometryAnalyzer::analyze( const Event& ev,
				       const EventSetup& es ) {

  ESHandle<MTDDetLayerGeometry> geo;
  es.get<MTDRecoGeometryRecord>().get(geo);

  ESHandle<MagneticField> magfield;
  es.get<IdealMagneticFieldRecord>().get(magfield);
  // Some printouts

  cout << "*** allBTLLayers(): " << geo->allBTLLayers().size() << endl;
  for (auto dl = geo->allBTLLayers().begin();
       dl != geo->allBTLLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allBTLLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allETLLayers(): " << geo->allETLLayers().size() << endl;
  for (auto dl = geo->allETLLayers().begin();
       dl != geo->allETLLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allETLLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;
    
  cout << "*** allLayers(): " << geo->allLayers().size() << endl;
  for (auto dl = geo->allLayers().begin();
       dl != geo->allLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;





  testBTLLayers(geo.product(),magfield.product());
  testETLLayers(geo.product(),magfield.product());
}


void MTDRecoGeometryAnalyzer::testBTLLayers(const MTDDetLayerGeometry* geo,const MagneticField* field) {

  const vector<const DetLayer*>& layers = geo->allBTLLayers();

  for (auto ilay = layers.begin(); ilay!=layers.end(); ++ilay) {
    const MTDTrayBarrelLayer* layer = (const MTDTrayBarrelLayer*) (*ilay);
  
    const BoundCylinder& cyl = layer->specificSurface();  

    double halfZ = cyl.bounds().length()/2.;

    // Generate a random point on the cylinder
    double aPhi = CLHEP::RandFlat::shoot(-Geom::pi(),Geom::pi());
    double aZ = CLHEP::RandFlat::shoot(-halfZ, halfZ);
    GlobalPoint gp(GlobalPoint::Cylindrical(cyl.radius(), aPhi, aZ));  

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp,gv,charge,field);
    TrajectoryStateOnSurface tsos(gtp, cyl);
    cout << "testBTLLayers: at " << tsos.globalPosition()
	 << " R=" << tsos.globalPosition().perp()
	 << " phi=" << tsos.globalPosition().phi()
	 << " Z=" << tsos.globalPosition().z()
	 << " p = " << tsos.globalMomentum()
	 << endl;


    SteppingHelixPropagator prop(field,anyDirection);

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
	   << " id: " << std::hex << BTLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec<< endl 
	   << "  distance " << (tsos.globalPosition()-compDets.front().first->position()).mag()

	   << endl
	   << endl;
    } else {
      cout << " ERROR : no compatible det found" << endl;
    }    
  }
}

void MTDRecoGeometryAnalyzer::testETLLayers(const MTDDetLayerGeometry* geo,const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  for (auto ilay = layers.begin(); ilay!=layers.end(); ++ilay) {
    const MTDRingForwardDoubleLayer* layer = (const MTDRingForwardDoubleLayer*) (*ilay);
  
    const BoundDisk& disk = layer->specificSurface();

    // Generate a random point on the disk
    double aPhi = CLHEP::RandFlat::shoot(-Geom::pi(),Geom::pi());
    double aR = CLHEP::RandFlat::shoot(disk.innerRadius(), disk.outerRadius());
    GlobalPoint gp(GlobalPoint::Cylindrical(aR, aPhi, disk.position().z()));  

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp,gv,charge,field);
    TrajectoryStateOnSurface tsos(gtp, disk);
    cout << "testETLLayers: at " << tsos.globalPosition()
	 << " R=" << tsos.globalPosition().perp()
	 << " phi=" << tsos.globalPosition().phi()
	 << " Z=" << tsos.globalPosition().z()
	 << " p = " << tsos.globalMomentum()
	 << endl;


    SteppingHelixPropagator prop(field,anyDirection);

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
	   << " id: " << std::hex << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec << endl 
	   << "  distance " << (tsos.globalPosition()-compDets.front().first->position()).mag()

	   << endl
	   << endl;
    } else {
      if(layer->isCrack(gp))
      {
         cout << " CSC crack found ";
      }
      else
      {
        cout << " ERROR : no compatible det found in CSC"
          << " at: R=" << gp.perp()
          << " phi= " << gp.phi().degrees()
          << " Z= " << gp.z();
      }

    }
  }
}

string MTDRecoGeometryAnalyzer::dumpLayer(const DetLayer* layer) const {
  stringstream output;
  
  const BoundSurface* sur=0;
  const BoundCylinder* bc=0;
  const BoundDisk* bd=0;

  sur = &(layer->surface());
  if ( (bc = dynamic_cast<const BoundCylinder*>(sur)) ) {
    output << "  Cylinder of radius: " << bc->radius() << endl;
  }
  else if ( (bd = dynamic_cast<const BoundDisk*>(sur)) ) {
    output << "  Disk at: " <<  bd->position().z() << endl;
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDRecoGeometryAnalyzer);
