/*
 * LaserSeedGenerator.cc --- Seeds for Tracking of Laser Beams
 */

#include "Alignment/LaserAlignment/plugins/LaserSeedGenerator.h" 

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

//
// constructors and destructor
//
LaserSeedGenerator::LaserSeedGenerator(const edm::ParameterSet& iConfig)
  : conf_(iConfig), laser_seed(iConfig)
{
  edm::LogInfo("LaserSeedGenerator") << " Entering the LaserSeedGenerator ";

  std::string alias ( iConfig.getParameter<std::string>("@module_label") );

  //register your products
  produces<TrajectorySeedCollection>().setBranchAlias( alias + "Seeds" );

  //now do what ever other initialization is needed

}


LaserSeedGenerator::~LaserSeedGenerator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to produce the data  ------------
void
LaserSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get the input
  edm::InputTag matchedRecHitsTag = conf_.getParameter<edm::InputTag>("matchedRecHits");
  edm::InputTag rphiRecHitsTag = conf_.getParameter<edm::InputTag>("rphiRecHits");
  edm::InputTag stereoRecHitsTag = conf_.getParameter<edm::InputTag>("stereoRecHits");

  edm::Handle<SiStripRecHit2DCollection> rphiRecHits;
  iEvent.getByLabel(rphiRecHitsTag, rphiRecHits);
  edm::Handle<SiStripRecHit2DCollection> stereoRecHits;
  iEvent.getByLabel(stereoRecHitsTag, stereoRecHits);
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHits;
  iEvent.getByLabel(matchedRecHitsTag, matchedRecHits);

  // create empty output collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

  // initialize the seed generator
  laser_seed.init(*stereoRecHits, *rphiRecHits, *matchedRecHits, iSetup);

  // invoke the seed finding algorithm
  laser_seed.run(*output,iSetup);

  // put the TrajectorySeedCollection in the event
  LogDebug("Algorithm Performance") << " number of seeds = " << output->size();
  iEvent.put(output);
}

// ------------ method called once each job just before starting event loop  ------------
void 
LaserSeedGenerator::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LaserSeedGenerator::endJob() 
{
}
// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserSeedGenerator);
