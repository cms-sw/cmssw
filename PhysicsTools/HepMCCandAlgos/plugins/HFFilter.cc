#include "PhysicsTools/HepMCCandAlgos/interface/HFFilter.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace std;

//
// constructors and destructor
//
HFFilter::HFFilter(const edm::ParameterSet& iConfig)
{
  genJetsCollName_     = iConfig.getParameter<edm::InputTag>("genJetsCollName");
  ptMin_               = iConfig.getParameter<double>("ptMin");
  etaMax_              = iConfig.getParameter<double>("etaMax");
}


HFFilter::~HFFilter()
{
}


//
// member functions
//

// Filter event based on whether there are heavy flavor GenJets in it that satisfy
// pt and eta cuts
bool
HFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the GenJetCollection
   using namespace edm;
   using namespace reco;
   Handle<std::vector<GenJet> > hGenJets;
   iEvent.getByLabel(genJetsCollName_,hGenJets);

   // Loop over the GenJetCollection
   vector<GenJet>::const_iterator ijet = hGenJets->begin();
   vector<GenJet>::const_iterator end  = hGenJets->end();
   for ( ; ijet != end; ++ijet ) {

     // Check to make sure the GenJet satisfies kinematic cuts. Ignore those that don't. 
     if ( ijet->pt() < ptMin_ || fabs(ijet->eta()) > etaMax_ ) continue;

     // Get the constituent particles
     vector<const GenParticle*> particles = ijet->getGenConstituents ();
    
     // Loop over the constituent particles
     vector<const GenParticle*>::const_iterator genit = particles.begin();
     vector<const GenParticle*>::const_iterator genend = particles.end();
     for ( ; genit != genend; ++genit ) {

       // See if any of them come from B or C hadrons
       const GenParticle & genitref = **genit;
       if ( JetMCTagUtils::decayFromBHadron( genitref ) ||
	    JetMCTagUtils::decayFromCHadron( genitref ) ) {
	 return true;
       }
     }// end loop over constituents
   }// end loop over jets


   return false;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HFFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFFilter);
