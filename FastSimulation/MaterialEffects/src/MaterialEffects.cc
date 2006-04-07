#include "TrackingTools/PatternTools/interface/MediumProperties.h"

#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"

//#include "FastSimulation/Utilitiess/interface/FamosHistos.h"

#include <list>
#include <utility>
#include <iomanip>

using namespace std;

MaterialEffects::MaterialEffects()
{
  // Set the minimal photon energy for a Brem from e+/-

  vector<int> defaultProcesses;

  defaultProcesses.push_back(1);  // Simulate Pair Production 
  defaultProcesses.push_back(1);  // Simulate Bremsstrahlung
  defaultProcesses.push_back(1);  // Simulate Energy Loss
  defaultProcesses.push_back(1);  // Simulate Multiple Scattering 

  //  ConfigurableVector<int> 
  //    theProcesses(defaultProcesses,"MaterialEffects:Options");

  //  if ( theProcesses.size() == 4 ) {
  //    doPairProduction     = theProcesses[0];
  //    doBremsstrahlung     = theProcesses[1];
  //    doEnergyLoss         = theProcesses[2];
  //    doMultipleScattering = theProcesses[3];
  //  } else {
    doPairProduction     = defaultProcesses[0];
    doBremsstrahlung     = defaultProcesses[1];
    doEnergyLoss         = defaultProcesses[2];
    doMultipleScattering = defaultProcesses[3];
    //  }
  
  // Set the minimal pT before giving up the dE/dx treatment
  double defaultCut = 0.050; // Lower pT (in GeV/c)
  //  SimpleConfigurable<double> theCut(defaultCut,"EnergyLoss:Cuts");
  //  pTmin = theCut.value();
  pTmin =defaultCut;
}


void MaterialEffects::interact(FSimEvent& mySimEvent, 
			       const TrackerLayer& layer,
			       ParticlePropagator& myTrack,
			       unsigned itrack) {

  MaterialEffectsUpdator::RHEP_const_iter DaughterIter;
  double radlen;

  /* For radiation length tuning */
  /* 
  FamosHistos* myHistos = FamosHistos::instance();

  bool plot = 
    ( myTrack.vect().mag() > 1.5 && 
      abs(myTrack.pid()) == 11 && 
      itrack < 2 ) ? 
    true : false;

  double radius = myTrack.position().perp()*0.1;
  double zed = fabs(myTrack.position().z())*0.1;

  if ( plot && radius < 2.6 ) { 
    myEta = myTrack.eta();
    myHistos->fill("h404",myEta);
  }
  */


//-------------------
//  Photon Conversion
//-------------------
  if ( doPairProduction && myTrack.pid()==22 ) {
    
    radlen = radLengths(layer,myTrack);
    PairProduction.updateState(myTrack,radlen);

    if ( PairProduction.nDaughters() ) {	
      //add a vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack);
      //Fill("h200",myTrack.vertex().z()*0.1,myTrack.vertex().perp()*0.1);
      
      // This was a photon that converted
      for ( DaughterIter = PairProduction.beginDaughters();
	    DaughterIter != PairProduction.endDaughters(); 
	    ++DaughterIter) {

	mySimEvent.addSimTrack(*DaughterIter, ivertex);

      }
      // The photon converted. Return.
      return;
    }
  } 

  if ( myTrack.charge() == 0 ) return;
  radlen = radLengths(layer,myTrack);

//----------------
//  Bremsstrahlung
//----------------

  if ( doBremsstrahlung && abs(myTrack.pid())==11 ) {
        
    Bremsstrahlung.updateState(myTrack,radlen);

    /* For radiation length tuning */
    /*
    if ( plot ) {
      if ( radius <  20. && zed <  70. ) myHistos->fill("h401",myEta,radlen);
      if ( radius <  55. && zed < 120. ) myHistos->fill("h402",myEta,radlen);
      if ( radius < 115. && zed < 280. ) myHistos->fill("h403",myEta,radlen);
      myHistos->fill("h400",myEta,radlen);
    }
    */

    if ( Bremsstrahlung.nDaughters() ) {
      
      // Add a vertex, but do not attach it to the electron, because it 
      // continues its way...
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),-1);
      //myHistos->fill("h200",myTrack.vertex().z()*0.1,myTrack.vertex().perp()*0.1);

      for ( DaughterIter = Bremsstrahlung.beginDaughters();
	    DaughterIter != Bremsstrahlung.endDaughters(); 
	    ++DaughterIter) {
	mySimEvent.addSimTrack(*DaughterIter, ivertex);
      }
      
    }
    
  } 


////--------------
////  Energy loss 
///---------------

  if ( doEnergyLoss )
    EnergyLoss.updateState(myTrack,radlen);

////----------------------
////  Multiple scattering
///-----------------------

  if ( doMultipleScattering && myTrack.perp() > pTmin ) {
    MultipleScattering.setNormalVector(normalVector(layer,myTrack));
    MultipleScattering.updateState(myTrack,radlen);
  }
    
}

double
MaterialEffects::radLengths(const TrackerLayer& layer,
			    ParticlePropagator& myTrack ) const {

  const Surface& surface = layer.surface();
  const MediumProperties& mp = *surface.mediumProperties();
  double radlen = mp.radLen();

  GlobalVector P(myTrack.px(),myTrack.py(),myTrack.pz());
  
  //  GlobalVector normal =   layer.forward() ?  
  //    ((BoundPlane*)&surface)->normalVector() : 
  //    GlobalVector(myTrack.x()/myTrack.vertex().perp(),
  //		 myTrack.y()/myTrack.vertex().perp(),
  //		 0.0);
  GlobalVector normal = normalVector(layer,myTrack);

  radlen /= fabs(P.dot(normal)/(P.mag()*normal.mag()));

  // This is disgusting. It should be in the geometry description, by there
  // is no way to define a cylinder with a hole in the middle...
  double rad = myTrack.vertex().perp();
  double zed = fabs(myTrack.vertex().z());
  if ( rad > 160. && zed < 280. ) {
    // Less material on all sensitive layers of the Silicon Tracker
    if ( rad > 550. && zed < 200. && layer.sensitive() ) radlen *= 0.50;
    // Much less cables outside the Si Tracker barrel
    if ( rad > 1180. ) radlen *= 0.2;
    // No cable whatsoever in the Pixel Barrel.
    if ( rad < 180. && zed < 260. ) radlen *= 0.04;
  }

  return radlen;

}

GlobalVector
MaterialEffects::normalVector(const TrackerLayer& layer,
			      ParticlePropagator& myTrack ) const {
  return layer.forward() ?  
    (dynamic_cast<const Plane*>(&(layer.surface())))->normalVector() : 
    GlobalVector(myTrack.x()/myTrack.vertex().perp(),
		 myTrack.y()/myTrack.vertex().perp(),
		 0.0);
}
