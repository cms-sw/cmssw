#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedGenerator.h"
#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

using namespace edm;
using namespace std;
using namespace reco;


//
// constructors and destructor
//
NuclearSeedGenerator::NuclearSeedGenerator(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
   produces<TrajectorySeedCollection>();
}


NuclearSeedGenerator::~NuclearSeedGenerator()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
NuclearSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   typedef TrajectoryMeasurement TM;

   LogDebug("NuclearSeedGenerator") << "Entering in produce" << "\n";

   edm::Handle<reco::TrackCollection> m_TrackCollection;
   iEvent.getByLabel( "TrackRefitter", m_TrackCollection );
   edm::Handle<std::vector<Trajectory> > m_TrajectoryCollection;
   iEvent.getByLabel( "TrackRefitter", m_TrajectoryCollection );

   LogDebug("NuclearSeedGenerator") << "Number of trajectory in event :" << m_TrajectoryCollection->size() << "\n";

   std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

   // Run the nuclear finder algorithm
   theNuclearInteractionFinder->setEvent(iEvent);

   std::vector<std::pair<TM, std::vector<TM> > >  tmPairs;
   if(!m_TrajectoryCollection->empty())
          tmPairs = theNuclearInteractionFinder->run( *(m_TrajectoryCollection.product())  );

   LogDebug("NuclearSeedGenerator") << "Size of the vector of pairs: " << tmPairs.size() << "\n";
   for(std::vector<std::pair<TM, std::vector<TM> > >::const_iterator itp = tmPairs.begin(); itp!=tmPairs.end(); itp++) {
         const TM& innerHit = itp->first;
         const std::vector<TM>& outerHits = itp->second;
         LogDebug("NuclearSeedGenerator") << "Size of the vector of outer hits : " << outerHits.size() << "\n";
         for(std::vector<TM>::const_iterator outhit = outerHits.begin(); outhit!=outerHits.end(); outhit++) {
               if((innerHit.recHit())->isValid() && (outhit->recHit())->isValid()) {
                     theSeed->setMeasurements(innerHit, *outhit);
                     if(theSeed->isValid()) {
                          TrajectorySeed ptraj = theSeed->TrajSeed();
                          //output->push_back(theSeed->TrajSeed());
                          output->push_back(ptraj);
                          LogDebug("NuclearSeedGenerator") << "Seed put in event: " << "\n"
                                                           << "Nhits : " << ptraj.nHits() << "\n"
                                                           << "Initial state : " << (ptraj.startingState()).parameters().position() << "\n";
                     }
                     else LogDebug("NuclearSeedGenerator") << "The seed is invalid" << "\n";
               }
               else  LogDebug("NuclearSeedGenerator") << "The initial hits for seeding are invalid" << "\n";
         }
   }
   iEvent.put(output);
}

// ------------ method called once each job just before starting event loop  ------------
void 
NuclearSeedGenerator::beginJob(const edm::EventSetup& es)
{
   theNuclearInteractionFinder = std::auto_ptr<NuclearInteractionFinder>(new NuclearInteractionFinder(es, conf_));
   theSeed = boost::shared_ptr<SeedFromNuclearInteraction>(new SeedFromNuclearInteraction(es, conf_));
}

void  NuclearSeedGenerator::endJob() {}


//define this as a plug-in
DEFINE_FWK_MODULE(NuclearSeedGenerator);
