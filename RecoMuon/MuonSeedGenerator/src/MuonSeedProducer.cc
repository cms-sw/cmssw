/**
 *  See header file for a description of this class.
 *  
 *  \author Dominique Fortin - UCR
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedProducer.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedBuilder.h"


// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

// For efficiency study
#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

// Magnetic Field
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/Common/interface/Handle.h>
// C++
#include <vector>



/* 
 * Constructor
 */
MuonSeedProducer::MuonSeedProducer(const edm::ParameterSet& pset){

  // Register what this produces
  produces<TrajectorySeedCollection>(); 

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugMuonSeed");

  // Builder which returns seed collection 
  muonSeedBuilder_   = new MuonSeedBuilder( pset ); 


  if (debug) {
    // Initialize statistics
    for ( int i = 0; i < 120; i++ ) {
      theNumerator[i]=0;
      theNumerator2[i]=0;
      theDenominator[i]=0;
    }
  }

}


/*
 * Destructor
 */
MuonSeedProducer::~MuonSeedProducer(){

  delete muonSeedBuilder_;


  if ( debug ) {
    // Printout efficiency performance and ghost rate
    std::cout << "*** Efficiency and ghost rate *** "<< std::endl;
    std::cout << "eta | # track | # reco seed | # ghost | eff" << std::endl;
    
    float eff, ghost;
    float sumSeed  = 0;
    float sumEvent = 0;
    float sumGhost = 0;
    for ( int i = 0; i < 120; i++ ) {
      sumSeed += 1.* theNumerator[i];
      sumEvent+= 1.* theDenominator[i];
      sumGhost+= 1.* theNumerator2[i];
      float eta = i * 2. / 100.;
      if ( theDenominator[i] > 0 ) {
	eff = (1.*theNumerator[i])/theDenominator[i];
	ghost = theNumerator2[i];
      } else {
	eff = -1;
	ghost = -1;
      }
      std::cout << eta << "  " << theDenominator[i] << "  " 
		<< theNumerator[i] << "  " << ghost << "  " << eff << std::endl;
    }
    std::cout << "Global eff and fake rate" << std::endl;
    if (sumEvent > 0 ) {
      std::cout << "Eff   = " << sumSeed/sumEvent << std::endl;
      std::cout << "Ghost = " << sumGhost/sumEvent << std::endl;
    }
  }

}


/*
 * Producer (the main)
 */ 
void MuonSeedProducer::produce(edm::Event& event, const edm::EventSetup& eSetup){

  // Muon Geometry
  edm::ESHandle<MuonDetLayerGeometry> muonLayers;
  eSetup.get<MuonRecoGeometryRecord>().get(muonLayers);
  const MuonDetLayerGeometry* lgeom = &*muonLayers;
  muonSeedBuilder_->setGeometry( lgeom );

  // Magnetic field
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);
  const MagneticField* theField = &*field;
  muonSeedBuilder_->setBField( theField );

  // Simtrack for efficiency study
  edm::Handle<edm::SimTrackContainer> simTracks;
  event.getByLabel("g4SimHits",simTracks);

  // Create pointer to the seed container

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection() );

  int nSeeds = 0;
  nSeeds = muonSeedBuilder_->build( event, eSetup, *output);

  // Append muon seed collection to event
  event.put( output );


  // This is a statistical test useful only for debugging purposes.
  // Obviously this should be remove when running on data
  if (debug) {
    // Simtrack for efficiency study
    edm::Handle<edm::SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
    
    int theEta = -1;
    for (edm::SimTrackContainer::const_iterator it = simTracks->begin(); it != simTracks->end(); ++it) {
      if (abs((*it).type()) != 13) continue;
      double Eta = (*it).momentum().eta();
      if (Eta < 0.) Eta = -Eta;
      theEta = int(Eta * 100. / 2.);
      break;
    }    
    
    if (theEta > -1 && theEta < 120) {
      theDenominator[theEta]++;
            
      if (nSeeds > 0) theNumerator[theEta]++;
      if (nSeeds > 1) theNumerator2[theEta]++;
    }
  }

}
