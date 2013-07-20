/** \file
 *
 *  $Date: 2009/05/27 08:09:45 $
 *  $Revision: 1.9 $
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
#include "RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"

#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <sstream>

#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace edm;

class MuonRecoGeometryAnalyzer : public EDAnalyzer {
 public:

  MuonRecoGeometryAnalyzer( const ParameterSet& pset);

  virtual void analyze( const Event& ev, const EventSetup& es);

  void testDTLayers(const MuonDetLayerGeometry*, const MagneticField* field);
  void testCSCLayers(const MuonDetLayerGeometry*, const MagneticField* field);

  string dumpLayer(const DetLayer* layer) const;

 private:
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
  // Some printouts

  cout << "*** allDTLayers(): " << geo->allDTLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allDTLayers().begin();
       dl != geo->allDTLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allDTLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allCSCLayers(): " << geo->allCSCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allCSCLayers().begin();
       dl != geo->allCSCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allCSCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** forwardCSCLayers(): " << geo->forwardCSCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->forwardCSCLayers().begin();
       dl != geo->forwardCSCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->forwardCSCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** backwardCSCLayers(): " << geo->backwardCSCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->backwardCSCLayers().begin();
       dl != geo->backwardCSCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->backwardCSCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allRPCLayers(): " << geo->allRPCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allRPCLayers().begin();
       dl != geo->allRPCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allRPCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** endcapRPCLayers(): " << geo->endcapRPCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->endcapRPCLayers().begin();
       dl != geo->endcapRPCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->endcapRPCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** barrelRPCLayers(): " << geo->barrelRPCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->barrelRPCLayers().begin();
       dl != geo->barrelRPCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->barrelRPCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** forwardRPCLayers(): " << geo->forwardRPCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->forwardRPCLayers().begin();
       dl != geo->forwardRPCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->forwardRPCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** backwardRPCLayers(): " << geo->backwardRPCLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->backwardRPCLayers().begin();
       dl != geo->backwardRPCLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->backwardRPCLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allBarrelLayers(): " << geo->allBarrelLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allBarrelLayers().begin();
       dl != geo->allBarrelLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allBarrelLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allEndcapLayers(): " << geo->allEndcapLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allEndcapLayers().begin();
       dl != geo->allEndcapLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allEndcapLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allForwardLayers(): " << geo->allForwardLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allForwardLayers().begin();
       dl != geo->allForwardLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allForwardLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;

  cout << "*** allBackwardLayers(): " << geo->allBackwardLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allBackwardLayers().begin();
       dl != geo->allBackwardLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allBackwardLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;


  cout << "*** allLayers(): " << geo->allLayers().size() << endl;
  for (vector<DetLayer*>::const_iterator dl = geo->allLayers().begin();
       dl != geo->allLayers().end(); ++dl) {
    cout << "  " << (int) (dl-geo->allLayers().begin()) << " " << dumpLayer(*dl);
  }
  cout << endl << endl;





  testDTLayers(geo.product(),magfield.product());
  testCSCLayers(geo.product(),magfield.product());
}


void MuonRecoGeometryAnalyzer::testDTLayers(const MuonDetLayerGeometry* geo,const MagneticField* field) {

  const vector<DetLayer*>& layers = geo->allDTLayers();

  for (vector<DetLayer*>::const_iterator ilay = layers.begin(); ilay!=layers.end(); ++ilay) {
    const MuRodBarrelLayer* layer = (const MuRodBarrelLayer*) (*ilay);
  
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
    cout << "testDTLayers: at " << tsos.globalPosition()
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
    const MuRingForwardDoubleLayer* layer = (const MuRingForwardDoubleLayer*) (*ilay);
  
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
    cout << "testCSCLayers: at " << tsos.globalPosition()
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
	   << " id: " << CSCDetId(compDets.front().first->geographicalId().rawId()) << endl 
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

string MuonRecoGeometryAnalyzer::dumpLayer(const DetLayer* layer) const {
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
DEFINE_FWK_MODULE(MuonRecoGeometryAnalyzer);
