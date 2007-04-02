#include "TrackingTools/PatternTools/interface/MediumProperties.h"

// Famos Headers
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"

#include "FastSimulation/MaterialEffects/interface/PairProductionSimulator.h"
#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/MaterialEffects/interface/BremsstrahlungSimulator.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
#include "FastSimulation/MaterialEffects/interface/NuclearInteractionSimulator.h"
#include "FastSimulation/MaterialEffects/interface/NuclearInteractionEDMSimulator.h"

#include <list>
#include <utility>
#include <iostream>
#include <string>

using namespace std;

MaterialEffects::MaterialEffects(const edm::ParameterSet& matEff,
				 const RandomEngine* engine)
  : PairProduction(0), Bremsstrahlung(0), 
    MultipleScattering(0), EnergyLoss(0), 
    NuclearInteraction(0), NuclearInteractionEDM(0), 
    pTmin(999.), random(engine)
{
  // Set the minimal photon energy for a Brem from e+/-

  bool doPairProduction     = matEff.getParameter<bool>("PairProduction");
  bool doBremsstrahlung     = matEff.getParameter<bool>("Bremsstrahlung");
  bool doEnergyLoss         = matEff.getParameter<bool>("EnergyLoss");
  bool doMultipleScattering = matEff.getParameter<bool>("MultipleScattering");
  bool doNuclearInteraction = matEff.getParameter<bool>("NuclearInteraction");
  bool doNuclearInteractionEDM = matEff.getParameter<bool>("NuclearInteractionEDM");
  
  // Set the minimal pT before giving up the dE/dx treatment

  if ( doPairProduction ) { 

    double photonEnergy = matEff.getParameter<double>("photonEnergy");
    PairProduction = new PairProductionSimulator(photonEnergy,
					       random);

  }

  if ( doBremsstrahlung ) { 

    double bremEnergy = matEff.getParameter<double>("bremEnergy");
    double bremEnergyFraction = matEff.getParameter<double>("bremEnergyFraction");
    Bremsstrahlung = new BremsstrahlungSimulator(bremEnergy,
					       bremEnergyFraction,
					       random);

  }

  if ( doEnergyLoss ) { 

    pTmin = matEff.getParameter<double>("pTmin");
    EnergyLoss = new EnergyLossSimulator(random);

  }

  if ( doMultipleScattering ) { 

    MultipleScattering = new MultipleScatteringSimulator(random);

  }

  if ( doNuclearInteraction ) { 
    std::vector<std::string> listOfFiles 
      = matEff.getUntrackedParameter<std::vector<std::string> >("fileNames");
    vector<double> pionEnergies 
      = matEff.getUntrackedParameter<std::vector<double> >("pionEnergies");
    vector<double> ratioRatio 
      = matEff.getUntrackedParameter<std::vector<double> >("ratioRatio");
    double pionEnergy 
      = matEff.getParameter<double>("pionEnergy");
    double lengthRatio 
      = matEff.getParameter<double>("lengthRatio");
    string inputFile 
      = matEff.getUntrackedParameter<std::string>("inputFile");
    // Construction
    NuclearInteraction = 
      new NuclearInteractionSimulator(listOfFiles,
				    pionEnergies,
				    pionEnergy,
				    lengthRatio,
				    ratioRatio,
				    inputFile,
				    random);
  }

  if ( doNuclearInteractionEDM ) { 
    edm::ParameterSet listOfEDMFiles 
      = matEff.getParameter<edm::ParameterSet>("NuclearInteractionInput");
    vector<double> pionEnergies 
      = matEff.getUntrackedParameter<std::vector<double> >("pionEnergies");
    double pionEnergy 
      = matEff.getParameter<double>("pionEnergy");
    double lengthRatio 
      = matEff.getParameter<double>("lengthRatio");
    NuclearInteractionEDM = 
      new NuclearInteractionEDMSimulator(listOfEDMFiles,
				       pionEnergies,
				       pionEnergy,
				       lengthRatio,
				       random);
  }

}


MaterialEffects::~MaterialEffects() {

  if ( PairProduction ) delete PairProduction;
  if ( Bremsstrahlung ) delete Bremsstrahlung;
  if ( EnergyLoss ) delete EnergyLoss;
  if ( MultipleScattering ) delete MultipleScattering;
  if ( NuclearInteraction ) delete NuclearInteraction;
  if ( NuclearInteractionEDM ) delete NuclearInteractionEDM;

}

void MaterialEffects::interact(FSimEvent& mySimEvent, 
			       const TrackerLayer& layer,
			       ParticlePropagator& myTrack,
			       unsigned itrack) {

  MaterialEffectsSimulator::RHEP_const_iter DaughterIter;
  double radlen;

  /* For radiation length tuning */
  /* 
  FamosHistos* myHistos = FamosHistos::instance();

  bool plot = 
    ( myTrack.vect().mag() > 1.5 && 
      abs(myTrack.pid()) == 11 && 
      itrack < 2 ) ? 
    true : false;

  double radius = myTrack.position().perp();
  double zed = fabs(myTrack.position().z());

  if ( plot && radius < 2.6 ) { 
    myEta = myTrack.eta();
    myHistos->fill("h404",myEta);
  }
  */


//-------------------
//  Photon Conversion
//-------------------

  if ( PairProduction && myTrack.pid()==22 ) {
    
    theNormalVector = normalVector(layer,myTrack);
    radlen = radLengths(layer,myTrack);
    //
    PairProduction->updateState(myTrack,radlen);

    if ( PairProduction->nDaughters() ) {	
      //add a vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack);
      //Fill("h200",myTrack.vertex().z(),myTrack.vertex().perp());
      
      // This was a photon that converted
      for ( DaughterIter = PairProduction->beginDaughters();
	    DaughterIter != PairProduction->endDaughters(); 
	    ++DaughterIter) {

	mySimEvent.addSimTrack(*DaughterIter, ivertex);

      }
      // The photon converted. Return.
      return;
    }
  }

  if ( myTrack.pid() == 22 ) return;
  theNormalVector = normalVector(layer,myTrack);
  radlen = radLengths(layer,myTrack);

//------------------------
//   Nuclear interactions
//------------------------ 

  if ( NuclearInteraction && abs(myTrack.pid()) > 22 ) { 

    // A few fudge factors ...
    double factor = 1.;

    if ( !layer.sensitive() ) { 
      if ( layer.layerNumber() == 107 ) { 
	double eta = myTrack.vertex().eta();
	factor = eta > 2.2 ? 1.0 +(eta-2.2)*3.0 : 1.0;
      }	else if ( layer.layerNumber() == 113 ) { 
	double zed = fabs(myTrack.vertex().z());
	factor = zed > 116. ? 0.6 : 1.4;
      } else if ( layer.layerNumber() == 115 ) {
	factor = 0.0;
      }
    }
    
    // Simulate a nuclear interaction
    NuclearInteraction->updateState(myTrack,radlen*factor);

    if ( NuclearInteraction->nDaughters() ) { 

      //add a end vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack);
      
      // This was a hadron that interacted inelastically
      for ( DaughterIter = NuclearInteraction->beginDaughters();
	    DaughterIter != NuclearInteraction->endDaughters(); 
	    ++DaughterIter) {

	mySimEvent.addSimTrack(*DaughterIter, ivertex);

      }
      // The hadron is destroyed. Return.
      return;
    }
    
  }

  if ( NuclearInteractionEDM && abs(myTrack.pid()) > 22 ) { 
    
    // A few fudge factors ...
    double factor = 1.;

    if ( !layer.sensitive() ) { 
      if ( layer.layerNumber() == 107 ) { 
	double eta = myTrack.vertex().eta();
	factor = eta > 2.2 ? 1.0 +(eta-2.2)*2.4 : 1.0;
      }	else if ( layer.layerNumber() == 113 ) { 
	double zed = fabs(myTrack.vertex().z());
	factor = zed > 116. ? 0.6 : 1.4;
      } else if ( layer.layerNumber() == 115 ) {
	factor = 0.0;
      }
    }

    // Simulate a nuclear interaction
    NuclearInteractionEDM->updateState(myTrack,radlen*factor);

    // Add a new vertex and daughters if inelastic scattering
    if ( NuclearInteractionEDM->nDaughters() ) { 

      //add a end vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack);
      
      // This was a hadron that interacted inelastically
      for ( DaughterIter = NuclearInteractionEDM->beginDaughters();
	    DaughterIter != NuclearInteractionEDM->endDaughters(); 
	    ++DaughterIter) {

	mySimEvent.addSimTrack(*DaughterIter, ivertex);

      }
      // The hadron is destroyed. Return.
      return;
    }
    
  }

  if ( myTrack.charge() == 0 ) return;

  if ( !Bremsstrahlung && !EnergyLoss && !MultipleScattering ) return;

//----------------
//  Bremsstrahlung
//----------------

  if ( Bremsstrahlung && abs(myTrack.pid())==11 ) {
        
    Bremsstrahlung->updateState(myTrack,radlen);

    /* For radiation length tuning */
    /*
    if ( plot ) {
      if ( radius <  20. && zed <  70. ) myHistos->fill("h401",myEta,radlen);
      if ( radius <  55. && zed < 120. ) myHistos->fill("h402",myEta,radlen);
      if ( radius < 115. && zed < 280. ) myHistos->fill("h403",myEta,radlen);
      myHistos->fill("h400",myEta,radlen);
    }
    */

    if ( Bremsstrahlung->nDaughters() ) {
      
      // Add a vertex, but do not attach it to the electron, because it 
      // continues its way...
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack);
       //myHistos->fill("h200",myTrack.vertex().z(),myTrack.vertex().perp());

      for ( DaughterIter = Bremsstrahlung->beginDaughters();
	    DaughterIter != Bremsstrahlung->endDaughters(); 
	    ++DaughterIter) {
	mySimEvent.addSimTrack(*DaughterIter, ivertex);
      }
      
    }
    
  } 


////--------------
////  Energy loss 
///---------------

  if ( EnergyLoss )
    EnergyLoss->updateState(myTrack,radlen);

////----------------------
////  Multiple scattering
///-----------------------

  if ( MultipleScattering && myTrack.perp() > pTmin ) {
    //    MultipleScattering->setNormalVector(normalVector(layer,myTrack));
    MultipleScattering->setNormalVector(theNormalVector);
    MultipleScattering->updateState(myTrack,radlen);
  }
    
}

double
MaterialEffects::radLengths(const TrackerLayer& layer,
			    ParticlePropagator& myTrack ) const {

  //  const Surface& surface = layer.surface();
  //  const MediumProperties& mp = *surface.mediumProperties();
  //  double radlen = mp.radLen();
  double radlen = layer.surface().mediumProperties()->radLen();

  GlobalVector P(myTrack.px(),myTrack.py(),myTrack.pz());
  
  //  GlobalVector normal =   layer.forward() ?  
  //    ((BoundPlane*)&surface)->normalVector() : 
  //    GlobalVector(myTrack.x()/myTrack.vertex().perp(),
  //		 myTrack.y()/myTrack.vertex().perp(),
  //		 0.0);
  //  GlobalVector normal = normalVector(layer,myTrack);

  //  radlen /= fabs(P.dot(normal)/(P.mag()*normal.mag()));
  radlen /= fabs(P.dot(theNormalVector)/(P.mag()*theNormalVector.mag()));

  // This is disgusting. It should be in the geometry description, by there
  // is no way to define a cylinder with a hole in the middle...
  double rad = myTrack.vertex().perp();
  double zed = fabs(myTrack.vertex().z());

  if ( rad > 16. && zed < 299. ) {

    // Simulate cables out the TEC's
    if ( zed > 122. && layer.sensitive() ) { 

      if ( zed < 165. ) { 
	if ( rad < 24. ) radlen *= 3.0;
      } else {
	if ( rad < 32.5 ) radlen *= 3.0;
	else if ( (zed > 220. && rad < 45.0) || 
		  (zed > 250. && rad < 54.) ) radlen *= 0.3;
      }
    }

    // Less material on all sensitive layers of the Silicon Tracker
    else if ( zed < 20. && layer.sensitive() ) { 
      if ( rad > 55. ) radlen *= 0.50;
      else if ( zed < 10 ) radlen *= 0.77;
    }
    // Much less cables outside the Si Tracker barrel
    else if ( rad > 118. && zed < 250. ) { 
      if ( zed < 116 ) radlen *= 0.225 * .75 ;
      else radlen *= .75;
    }
    // No cable whatsoever in the Pixel Barrel.
    else if ( rad < 18. && zed < 26. ) radlen *= 0.08;
  }

  return radlen;

}

GlobalVector
MaterialEffects::normalVector(const TrackerLayer& layer,
			      ParticlePropagator& myTrack ) const {
  return layer.forward() ?  
    //    (dynamic_cast<const Plane*>(&(layer.surface())))->normalVector() : 
    layer.disk()->normalVector() :
    GlobalVector(myTrack.x(),myTrack.y(),0.)/myTrack.vertex().perp();
  //		 myTrack.y()/myTrack.vertex().perp(),
  //		 0.0);
}

void 
MaterialEffects::save() { 

  // Save current nuclear interactions in the event libraries.
  if ( NuclearInteraction ) NuclearInteraction->save();

}
