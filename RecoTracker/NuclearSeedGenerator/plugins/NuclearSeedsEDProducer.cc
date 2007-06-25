#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsEDProducer.h"
#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
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
improveSeeds(iConfig.getParameter<bool>("improveSeeds"))
{
   produces<TrajectorySeedCollection>();
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

   edm::Handle<reco::TrackCollection> m_TrackCollection;
   iEvent.getByLabel( "TrackRefitter", m_TrackCollection );
   edm::Handle<std::vector<Trajectory> > m_TrajectoryCollection;
   iEvent.getByLabel( "TrackRefitter", m_TrajectoryCollection );

   LogDebug("NuclearSeedGenerator") << "Number of trajectory in event :" << m_TrajectoryCollection->size() << "\n";

   std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

   // Update the measurement 
   theNuclearInteractionFinder->setEvent(iEvent);

   for(std::vector<Trajectory>::const_iterator iTraj = m_TrajectoryCollection->begin(); iTraj != m_TrajectoryCollection->end(); iTraj++) {

         // run the finder
         theNuclearInteractionFinder->run( *iTraj );

         // improve seeds
         if( improveSeeds == true ) theNuclearInteractionFinder->improveSeeds();

         // push back the new persistent seeds in output
         theNuclearInteractionFinder->getPersistentSeeds( output );
   }

   iEvent.put(output);
}

// ------------ method called once each job just before starting event loop  ------------
void 
NuclearSeedsEDProducer::beginJob(const edm::EventSetup& es)
{
   theNuclearInteractionFinder = std::auto_ptr<NuclearInteractionFinder>(new NuclearInteractionFinder(es, conf_));
}

void  NuclearSeedsEDProducer::endJob() {}


