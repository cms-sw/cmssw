// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/UERegionSelector.h>
 
using namespace edm;
using namespace std;
using namespace reco;

UERegionSelector::UERegionSelector(const ParameterSet& pset )
{
   produces< TrackCollection >();

   // get collections from parameter set
   jetCollName   = pset.getUntrackedParameter<InputTag>("JetCollectionName"     ,std::string(""));
   //   particleCollName = pset.getUntrackedParameter<InputTag>("ParticleCollectionName",std::string(""));
   trackCollName = pset.getUntrackedParameter<InputTag>("TrackCollectionName",std::string(""));

   // get phi-range from parameter set
   deltaPhiByPiMinJetParticle = pset.getParameter<double>("DeltaPhiByPiMinJetParticle");
   deltaPhiByPiMaxJetParticle = pset.getParameter<double>("DeltaPhiByPiMaxJetParticle");
   deltaPhiByPiMinJetParticle = deltaPhiByPiMinJetParticle * TMath::Pi();
   deltaPhiByPiMaxJetParticle = deltaPhiByPiMaxJetParticle * TMath::Pi();
}


UERegionSelector::~UERegionSelector()
{
}


void UERegionSelector::produce(Event& e, const EventSetup& )
{
  LogDebug("") << "UERegionSelector::produce(Event& e, const EventSetup& )";

  // ===== declare View<Candidate> to be filled with the Candidates in
  //       requested phi-region around leading jet
  auto_ptr< TrackCollection > ueRegionTracks( new TrackCollection );
  //  ueRegionCandidates->reserve( 10 );
  ueRegionTracks->reserve( 10 );

  // ===== loop over jets
  if ( e.getByLabel( jetCollName, jetHandle ) )
    {
      if ( jetHandle->size() )
	{
	  //	  if ( e.getByLabel( particleCollName , particleHandle ) )
	  if ( e.getByLabel( trackCollName , trackHandle ) )
	    {
	      if ( trackHandle->size() )
		{
		  View<Candidate>::const_iterator jetIt( jetHandle->begin() ); // jets already ordered in pT
		  const Candidate *jet( &(*jetIt) );
		  
		  LogDebug("") << "UERegionSelector: Leading jet (pT, eta, phi) = ("
			       << jet->pt() << ", "
			       << jet->eta() << ", "
			       << jet->phi() << ")" << endl;

		  unsigned int itrack(0);

		  // ===== loop over particles
// 		  for( View<Candidate>::const_iterator it(particleHandle->begin()), 
// 			 itEnd(particleHandle->end());
		  for( View<Track>::const_iterator it(trackHandle->begin()), 
			 itEnd(trackHandle->end());
		       it != itEnd; ++it )
		    {
		      //		      const Candidate *particle( &(*it) );
		      const Track *track( &(*it) );
		      double deltaphi( TMath::Abs( deltaPhi( jet->phi(), track->phi() ) ) );
		      
		      LogDebug("") << "UERegionSelector: Particle (pT, eta, phi) = ("
				   << track->pt() << ", "
				   << track->eta() << ", "
				   << track->phi() << "), "
				   << "dphi = " << deltaphi << endl;

		      // ===== check if particle is in selected region
		      if ( deltaphi >= deltaPhiByPiMinJetParticle &&
			   deltaphi <  deltaPhiByPiMaxJetParticle )
			{
			  //			  auto_ptr<Candidate> ptr( particle->clone() );
			  Track *newTrack( new Track(*track) );

			  ueRegionTracks->push_back( *newTrack );

// 			  cout << "TEllipse* ellipse" << itrack << " = getEllipse( ";
// 			  cout << track->pt() << ", ";
// 			  cout << track->phi() << ", ";
// 			  cout << track->charge() << ", alpha); ellipse" << itrack << "->Draw(\"same\");\n";

			  ++itrack;
			}
		    }
		}
	    }
	}
    }

  LogDebug("") << "UERegionSelector: found " << ueRegionTracks->size() 
	       << " Candidates in selected phi-region around leading jet" << endl;

  e.put( ueRegionTracks );
 
}

void UERegionSelector::beginJob(const edm::EventSetup&)
{
}

void UERegionSelector::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(UERegionSelector);
