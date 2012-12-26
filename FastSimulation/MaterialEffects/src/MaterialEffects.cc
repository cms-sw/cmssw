
//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//TrackingTools Headers

// Famos Headers
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"

#include "FastSimulation/MaterialEffects/interface/PairProductionSimulator.h"
#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/MaterialEffects/interface/BremsstrahlungSimulator.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
#include "FastSimulation/MaterialEffects/interface/NuclearInteractionSimulator.h"
#include "FastSimulation/MaterialEffects/interface/MuonBremsstrahlungSimulator.h"

#include <list>
#include <map>
#include <string>

MaterialEffects::MaterialEffects(const edm::ParameterSet& matEff,
				 const RandomEngine* engine)
  : PairProduction(0), Bremsstrahlung(0),MuonBremsstrahlung(0),
    MultipleScattering(0), EnergyLoss(0), 
    NuclearInteraction(0),
    pTmin(999.), random(engine)
{
  // Set the minimal photon energy for a Brem from e+/-

  bool doPairProduction     = matEff.getParameter<bool>("PairProduction");
  bool doBremsstrahlung     = matEff.getParameter<bool>("Bremsstrahlung");
  bool doEnergyLoss         = matEff.getParameter<bool>("EnergyLoss");
  bool doMultipleScattering = matEff.getParameter<bool>("MultipleScattering");
  bool doNuclearInteraction = matEff.getParameter<bool>("NuclearInteraction");
  bool doMuonBremsstrahlung = matEff.getParameter<bool>("MuonBremsstrahlung");

  double A = matEff.getParameter<double>("A");
  double Z = matEff.getParameter<double>("Z");
  double density = matEff.getParameter<double>("Density");
  double radLen = matEff.getParameter<double>("RadiationLength");
  
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
//muon Brem+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 if ( doMuonBremsstrahlung ) {

    double bremEnergy = matEff.getParameter<double>("bremEnergy");
    double bremEnergyFraction = matEff.getParameter<double>("bremEnergyFraction");
    MuonBremsstrahlung = new MuonBremsstrahlungSimulator(random,A,Z,density,radLen,bremEnergy,
                                                 bremEnergyFraction);

  }


 //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if ( doEnergyLoss ) { 

    pTmin = matEff.getParameter<double>("pTmin");
    EnergyLoss = new EnergyLossSimulator(random,A,Z,density,radLen);

  }

  if ( doMultipleScattering ) { 
    
    MultipleScattering = new MultipleScatteringSimulator(random,A,Z,density,radLen);

  }

  if ( doNuclearInteraction ) { 

    // The energies simulated
    std::vector<double> pionEnergies 
      = matEff.getUntrackedParameter<std::vector<double> >("pionEnergies");

    // The particle types simulated
    std::vector<int> pionTypes 
      = matEff.getUntrackedParameter<std::vector<int> >("pionTypes");

    // The corresponding particle names
    std::vector<std::string> pionNames 
      = matEff.getUntrackedParameter<std::vector<std::string> >("pionNames");

    // The corresponding particle masses
    std::vector<double> pionMasses 
      = matEff.getUntrackedParameter<std::vector<double> >("pionMasses");

    // The smallest momentum for inelastic interactions
    std::vector<double> pionPMin 
      = matEff.getUntrackedParameter<std::vector<double> >("pionMinP");

    // The interaction length / radiation length ratio for each particle type
    std::vector<double> lengthRatio 
      = matEff.getParameter<std::vector<double> >("lengthRatio");
    //    std::map<int,double> lengthRatio;
    //    for ( unsigned i=0; i<theLengthRatio.size(); ++i )
    //      lengthRatio[ pionTypes[i] ] = theLengthRatio[i];

    // A global fudge factor for TEC layers (which apparently do not react to 
    // hadrons the same way as all other layers...
    theTECFudgeFactor = matEff.getParameter<double>("fudgeFactor");

    // The evolution of the interaction lengths with energy
    std::vector<double> theRatios  
      = matEff.getUntrackedParameter<std::vector<double> >("ratios");
    //std::map<int,std::vector<double> > ratios;
    //for ( unsigned i=0; i<pionTypes.size(); ++i ) { 
    //  for ( unsigned j=0; j<pionEnergies.size(); ++j ) { 
    //	ratios[ pionTypes[i] ].push_back(theRatios[ i*pionEnergies.size() + j ]);
    //  }
    //}
    std::vector< std::vector<double> > ratios;
    ratios.resize(pionTypes.size());
    for ( unsigned i=0; i<pionTypes.size(); ++i ) { 
      for ( unsigned j=0; j<pionEnergies.size(); ++j ) { 
	ratios[i].push_back(theRatios[ i*pionEnergies.size() + j ]);
      }
    }

    // The smallest momentum for elastic interactions
    double pionEnergy 
      = matEff.getParameter<double>("pionEnergy");

    // The algorithm to compute the distance between primary and secondaries
    // when a nuclear interaction occurs
    unsigned distAlgo 
      = matEff.getParameter<unsigned>("distAlgo");
    double distCut 
      = matEff.getParameter<double>("distCut");

    // The file to read the starting interaction in each files
    // (random reproducibility in case of a crash)
    std::string inputFile 
      = matEff.getUntrackedParameter<std::string>("inputFile");

    // Build the ID map (i.e., what is to be considered as a proton, etc...)
    std::map<int,int> idMap;
    // Protons
    std::vector<int> idProtons 
      = matEff.getUntrackedParameter<std::vector<int> >("protons");
    for ( unsigned i=0; i<idProtons.size(); ++i ) 
      idMap[idProtons[i]] = 2212;
    // Anti-Protons
    std::vector<int> idAntiProtons 
      = matEff.getUntrackedParameter<std::vector<int> >("antiprotons");
    for ( unsigned i=0; i<idAntiProtons.size(); ++i ) 
      idMap[idAntiProtons[i]] = -2212;
    // Neutrons
    std::vector<int> idNeutrons 
      = matEff.getUntrackedParameter<std::vector<int> >("neutrons");
    for ( unsigned i=0; i<idNeutrons.size(); ++i ) 
      idMap[idNeutrons[i]] = 2112;
    // Anti-Neutrons
    std::vector<int> idAntiNeutrons 
      = matEff.getUntrackedParameter<std::vector<int> >("antineutrons");
    for ( unsigned i=0; i<idAntiNeutrons.size(); ++i ) 
      idMap[idAntiNeutrons[i]] = -2112;
    // K0L's
    std::vector<int> idK0Ls 
      = matEff.getUntrackedParameter<std::vector<int> >("K0Ls");
    for ( unsigned i=0; i<idK0Ls.size(); ++i ) 
      idMap[idK0Ls[i]] = 130;
    // K+'s
    std::vector<int> idKplusses 
      = matEff.getUntrackedParameter<std::vector<int> >("Kplusses");
    for ( unsigned i=0; i<idKplusses.size(); ++i ) 
      idMap[idKplusses[i]] = 321;
    // K-'s
    std::vector<int> idKminusses 
      = matEff.getUntrackedParameter<std::vector<int> >("Kminusses");
    for ( unsigned i=0; i<idKminusses.size(); ++i ) 
      idMap[idKminusses[i]] = -321;
    // pi+'s
    std::vector<int> idPiplusses 
      = matEff.getUntrackedParameter<std::vector<int> >("Piplusses");
    for ( unsigned i=0; i<idPiplusses.size(); ++i ) 
      idMap[idPiplusses[i]] = 211;
    // pi-'s
    std::vector<int> idPiminusses 
      = matEff.getUntrackedParameter<std::vector<int> >("Piminusses");
    for ( unsigned i=0; i<idPiminusses.size(); ++i ) 
      idMap[idPiminusses[i]] = -211;

    // Construction
    NuclearInteraction = 
      new NuclearInteractionSimulator(pionEnergies, pionTypes, pionNames, 
				      pionMasses, pionPMin, pionEnergy, 
				      lengthRatio, ratios, idMap, 
				      inputFile, distAlgo, distCut, random);
  }

}


MaterialEffects::~MaterialEffects() {

  if ( PairProduction ) delete PairProduction;
  if ( Bremsstrahlung ) delete Bremsstrahlung;
  if ( EnergyLoss ) delete EnergyLoss;
  if ( MultipleScattering ) delete MultipleScattering;
  if ( NuclearInteraction ) delete NuclearInteraction;
//Muon Brem
  if ( MuonBremsstrahlung ) delete MuonBremsstrahlung;
}

void MaterialEffects::interact(FSimEvent& mySimEvent, 
			       const TrackerLayer& layer,
			       ParticlePropagator& myTrack,
			       unsigned itrack) {

  MaterialEffectsSimulator::RHEP_const_iter DaughterIter;
  double radlen;
  theEnergyLoss = 0;
  theNormalVector = normalVector(layer,myTrack);
  radlen = radLengths(layer,myTrack);

//-------------------
//  Photon Conversion
//-------------------

  if ( PairProduction && myTrack.pid()==22 ) {
    
    //
    PairProduction->updateState(myTrack,radlen);

    if ( PairProduction->nDaughters() ) {	
      //add a vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack,
					    FSimVertexType::PAIR_VERTEX);
      
      // Check if it is a valid vertex first:
      if (ivertex>=0) {
	// This was a photon that converted
	for ( DaughterIter = PairProduction->beginDaughters();
	      DaughterIter != PairProduction->endDaughters(); 
	      ++DaughterIter) {
	  
	  mySimEvent.addSimTrack(&(*DaughterIter), ivertex);
	  
	}
	// The photon converted. Return.
	return;
      }
      else {
	edm::LogWarning("MaterialEffects") <<  " WARNING: A non valid vertex was found in photon conv. -> " << ivertex << std::endl;    
      }

    }

  }

  if ( myTrack.pid() == 22 ) return;

//------------------------
//   Nuclear interactions
//------------------------ 

  if ( NuclearInteraction && abs(myTrack.pid()) > 100 
                          && abs(myTrack.pid()) < 1000000) { 

    // Simulate a nuclear interaction
    double factor = 1.0;
    if (layer.layerNumber() >= 19 && layer.layerNumber() <= 27 ) 
      factor = theTECFudgeFactor;
    NuclearInteraction->updateState(myTrack,radlen*factor);

    if ( NuclearInteraction->nDaughters() ) { 

      //add a end vertex to the mother particle
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack,
					    FSimVertexType::NUCL_VERTEX);
      
      // Check if it is a valid vertex first:
      if (ivertex>=0) {
	// This was a hadron that interacted inelastically
	int idaugh = 0;
	for ( DaughterIter = NuclearInteraction->beginDaughters();
	      DaughterIter != NuclearInteraction->endDaughters(); 
	      ++DaughterIter) {
	  
	  // The daughter in the event
	  int daughId = mySimEvent.addSimTrack(&(*DaughterIter), ivertex);
	  
	  // Store the closest daughter in the mother info (for later tracking purposes)
	  if ( NuclearInteraction->closestDaughterId() == idaugh++ ) {
	    if ( mySimEvent.track(itrack).vertex().position().Pt() < 4.0 ) 
	      mySimEvent.track(itrack).setClosestDaughterId(daughId);
	  }
	  
	}
	// The hadron is destroyed. Return.
	return;
      }
      else {
	edm::LogWarning("MaterialEffects") <<  " WARNING: A non valid vertex was found in nucl. int. -> " << ivertex << std::endl;    
      }

    }
    
  }

  if ( myTrack.charge() == 0 ) return;

  if ( !MuonBremsstrahlung && !Bremsstrahlung && !EnergyLoss && !MultipleScattering ) return;

//----------------
//  Bremsstrahlung
//----------------

  if ( Bremsstrahlung && abs(myTrack.pid())==11 ) {
        
    Bremsstrahlung->updateState(myTrack,radlen);

    if ( Bremsstrahlung->nDaughters() ) {
      
      // Add a vertex, but do not attach it to the electron, because it 
      // continues its way...
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack,
					    FSimVertexType::BREM_VERTEX);

      // Check if it is a valid vertex first:
      if (ivertex>=0) {
	for ( DaughterIter = Bremsstrahlung->beginDaughters();
	      DaughterIter != Bremsstrahlung->endDaughters(); 
	      ++DaughterIter) {
	  mySimEvent.addSimTrack(&(*DaughterIter), ivertex);
	}
      }
      else {
	edm::LogWarning("MaterialEffects") <<  " WARNING: A non valid vertex was found in brem -> " << ivertex << std::endl;    
      }
      
    }
    
  } 

//---------------------------
//  Muon_Bremsstrahlung
//--------------------------

  if (  MuonBremsstrahlung && abs(myTrack.pid())==13 ) {
       
    MuonBremsstrahlung->updateState(myTrack,radlen);

    if ( MuonBremsstrahlung->nDaughters() ) {

      // Add a vertex, but do not attach it to the muon, because it 
      // continues its way...
      int ivertex = mySimEvent.addSimVertex(myTrack.vertex(),itrack,
                                            FSimVertexType::BREM_VERTEX);
 
     // Check if it is a valid vertex first:
      if (ivertex>=0) {
	for ( DaughterIter = MuonBremsstrahlung->beginDaughters();
	      DaughterIter != MuonBremsstrahlung->endDaughters();
	      ++DaughterIter) {
	  mySimEvent.addSimTrack(&(*DaughterIter), ivertex);
	}
      }
      else {
	edm::LogWarning("MaterialEffects") <<  " WARNING: A non valid vertex was found in muon brem -> " << ivertex << std::endl;    
      }

    }

  }

////--------------
////  Energy loss 
///---------------

  if ( EnergyLoss )
  {
    theEnergyLoss = myTrack.E();
    EnergyLoss->updateState(myTrack,radlen);
    theEnergyLoss -= myTrack.E();
  }
  

////----------------------
////  Multiple scattering
///-----------------------

  if ( MultipleScattering && myTrack.Pt() > pTmin ) {
    //    MultipleScattering->setNormalVector(normalVector(layer,myTrack));
    MultipleScattering->setNormalVector(theNormalVector);
    MultipleScattering->updateState(myTrack,radlen);
  }
    
}

double
MaterialEffects::radLengths(const TrackerLayer& layer,
			    ParticlePropagator& myTrack) {

  // Thickness of layer
  theThickness = layer.surface().mediumProperties().radLen();

  GlobalVector P(myTrack.Px(),myTrack.Py(),myTrack.Pz());
  
  // Effective length of track inside layer (considering crossing angle)
  //  double radlen = theThickness / fabs(P.dot(theNormalVector)/(P.mag()*theNormalVector.mag()));
  double radlen = theThickness / fabs(P.dot(theNormalVector)) * P.mag();

  // This is a series of fudge factors (from the geometry description), 
  // to describe the layer inhomogeneities (services, cables, supports...)
  double rad = myTrack.R();
  double zed = fabs(myTrack.Z());

  double factor = 1;

  // Are there fudge factors for this layer
  if ( layer.fudgeNumber() ) 

    // If yes, loop on them
    for ( unsigned int iLayer=0; iLayer < layer.fudgeNumber(); ++iLayer ) { 

      // Apply to R if forward layer, to Z if barrel layer
      if ( (  layer.forward() && layer.fudgeMin(iLayer) < rad && rad < layer.fudgeMax(iLayer) )  ||
	   ( !layer.forward() && layer.fudgeMin(iLayer) < zed && zed < layer.fudgeMax(iLayer) ) ) {
	factor = layer.fudgeFactor(iLayer);
	break;
      }
    
  }

  theThickness *= factor;

  return radlen * factor;

}

GlobalVector
MaterialEffects::normalVector(const TrackerLayer& layer,
			      ParticlePropagator& myTrack ) const {
  return layer.forward() ?  
    layer.disk()->normalVector() :
    GlobalVector(myTrack.X(),myTrack.Y(),0.)/myTrack.R();
}

void 
MaterialEffects::save() { 

  // Save current nuclear interactions in the event libraries.
  if ( NuclearInteraction ) NuclearInteraction->save();

}
