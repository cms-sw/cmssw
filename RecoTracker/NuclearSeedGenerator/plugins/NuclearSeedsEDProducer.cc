#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsEDProducer.h"
#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

using namespace edm;
using namespace std;
using namespace reco;


//
// constructors and destructor
//
NuclearSeedsEDProducer::NuclearSeedsEDProducer(const edm::ParameterSet& iConfig) : conf_(iConfig),
improveSeeds(iConfig.getParameter<bool>("improveSeeds")),
producer_(iConfig.getParameter<std::string>("producer"))
{
   produces<TrajectorySeedCollection>();
   produces<TrajectoryToSeedsMap>();


}


NuclearSeedsEDProducer::~NuclearSeedsEDProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
NuclearSeedsEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   typedef TrajectoryMeasurement TM;

   edm::Handle< TrajectoryCollection > m_TrajectoryCollection;
   iEvent.getByLabel( producer_, m_TrajectoryCollection );

   LogDebug("NuclearSeedGenerator") << "Number of trajectory in event :" << m_TrajectoryCollection->size() << "\n";

   std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
   std::auto_ptr<TrajectoryToSeedsMap> outAssoc(new TrajectoryToSeedsMap);

   // Update the measurement
   theNuclearInteractionFinder->setEvent(iEvent);
   NavigationSetter setter( *(theNuclearInteractionFinder->nav()) );

   std::vector<std::pair<int, int> > assocPair;
   int i=0;

   for(std::vector<Trajectory>::const_iterator iTraj = m_TrajectoryCollection->begin(); iTraj != m_TrajectoryCollection->end(); iTraj++,i++) {

         // run the finder
         theNuclearInteractionFinder->run( *iTraj );

         // improve seeds
         if( improveSeeds == true ) theNuclearInteractionFinder->improveSeeds();

         // push back the new persistent seeds in output
         std::auto_ptr<TrajectorySeedCollection> newSeeds(theNuclearInteractionFinder->getPersistentSeeds());
         output->insert(output->end(), newSeeds->begin(), newSeeds->end());

         // fill the id of the Trajectory and the if of the seed in assocPair
         for(unsigned int j=0; j<newSeeds->size(); j++) {
                  assocPair.push_back( std::make_pair( i, output->size()-newSeeds->size()+j ) );
         }

   }

   const edm::OrphanHandle<TrajectorySeedCollection> refprodTrajSeedColl = iEvent.put(output);

   for(std::vector<std::pair<int, int> >::const_iterator iVecP = assocPair.begin(); iVecP != assocPair.end(); iVecP++) {
        outAssoc->insert(edm::Ref<TrajectoryCollection>(m_TrajectoryCollection,iVecP->first), edm::Ref<TrajectorySeedCollection>(refprodTrajSeedColl, iVecP->second));
   }
   iEvent.put(outAssoc);

}

// ------------ method called once each job just before starting event loop  ------------
void
NuclearSeedsEDProducer::beginRun(edm::Run const& run, const edm::EventSetup& es)
{
   theNuclearInteractionFinder = std::auto_ptr<NuclearInteractionFinder>(new NuclearInteractionFinder(es, conf_));

}

void  NuclearSeedsEDProducer::endJob() {}


