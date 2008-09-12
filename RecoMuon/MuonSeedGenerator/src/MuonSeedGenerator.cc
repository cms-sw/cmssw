/**
 *  See header file for a description of this class.
 *  
 *  All the code is under revision
 *
 *  $Date: 2008/08/25 22:04:28 $
 *  $Revision: 1.24 $
 *
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author ported by: R. Bellan - INFN Torino
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedGenerator.h"

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedOrcaPatternRecognition.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedSimpleCleaner.h"


// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/Common/interface/Handle.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

// C++
#include <vector>

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

// Constructor
MuonSeedGenerator::MuonSeedGenerator(const edm::ParameterSet& pset)
: thePatternRecognition(new MuonSeedOrcaPatternRecognition(pset)),
  theSeedFinder(new MuonSeedFinder(pset)),
  theSeedCleaner(new MuonSeedSimpleCleaner())
{
  produces<TrajectorySeedCollection>(); 
}

// Destructor
MuonSeedGenerator::~MuonSeedGenerator(){
  delete thePatternRecognition;
  delete theSeedFinder;
  delete theSeedCleaner;
}


// reconstruct muon's seeds
void MuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup)
{
  // create the pointer to the Seed container
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);
  
  theSeedFinder->setBField(&*field);

  std::vector<MuonRecHitContainer> patterns;
  thePatternRecognition->produce(event, eSetup, patterns);

  for(std::vector<MuonRecHitContainer>::const_iterator seedSegments = patterns.begin();
      seedSegments != patterns.end(); ++seedSegments)
  {
    theSeedFinder->seeds(*seedSegments, *output);
  }

  theSeedCleaner->clean(*output);

  event.put(output);
}

  
