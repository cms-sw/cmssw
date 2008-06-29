#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include <iterators>

using namespace std;
using namespace reco;
using namespace edm;


ostream & operator<<( ostream & out, Candidate const & cand) 
{
  char buff[1000];
  sprintf(buff, "%5d, status = %5d, nda = %5d, pt = %6.2f, eta = %6.2f, phi = %6.2f, m = %6.2f", 
	  cand.pdgId(), cand.status(), cand.numberOfDaughters(),
	  cand.pt(), cand.eta(), cand.phi(), cand.mass() );
  out << buff;
  return out;
}

FlavorHistoryProducer::FlavorHistoryProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ),
  pdgIdToSelect_( p.getParameter<int> ("pdgIdToSelect") ),
  ptMinParticle_( p.getParameter<double>( "ptMinParticle") ),  
  ptMinShower_( p.getParameter<double>( "ptMinShower") ),  
  etaMaxParticle_( p.getParameter<double>( "etaMaxParticle" )),  
  etaMaxShower_( p.getParameter<double>( "etaMaxShower" )),
  flavorHistoryName_( p.getParameter<string>("flavorHistoryName") ),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) )
{
  produces<ValueMap<FlavorHistory> >(flavorHistoryName_);
}

FlavorHistoryProducer::~FlavorHistoryProducer() { 
}

void FlavorHistoryProducer::beginJob( const EventSetup & es ) {
  ;
}

void FlavorHistoryProducer::produce( Event& evt, const EventSetup& ) 
{
  
  // Get a handle to the particle collection (OwnVector)
  Handle<CandidateView > particlesViewH;
  evt.getByLabel( src_, particlesViewH );

//   const vector<Candidate const *> & particles = *particlesViewH;

  // Copy the View to an vector for easier iterator manipulation convenience
  vector<const Candidate* > particles;
  for( CandidateView::const_iterator p = particlesViewH->begin();  p != particlesViewH->end(); ++p ) {
    particles.push_back(&*p);
  }
  
  // Make a new flavor history vector
  auto_ptr<ValueMap<FlavorHistory> > flavorHistory ( new ValueMap<FlavorHistory> () ) ;

  vector<FlavorHistory> flavorHistoryVector;

  // ------------------------------------------------------------
  // Loop over partons
  // ------------------------------------------------------------
  vector<const Candidate* >::size_type j;
  vector<const Candidate* >::size_type j_max = particles.size();
  for( j=0; j<j_max; ++j ) {

    // Get the candidate
    const Candidate *p = particles[j];
    // Set up indices that we'll need for the flavor history
    vector<Candidate const *>::size_type partonIndex=j;
    vector<Candidate const *>::size_type progenitorIndex=0;
    vector<Candidate const *>::size_type sisterIndex=0;
    FlavorHistory::FLAVOR_T flavorSource=FlavorHistory::FLAVOR_NULL;


    int idabs = abs( (p)->pdgId() );
    int nDa = (p)->numberOfDaughters();

    // Check if we have a status 2 or 3 particle, which is a parton before the string
    if ( p->status() > 1 ) {
      if(verbose_) cout << "--------------------------" << endl;
      if(verbose_) cout << "Processing particle " << j  << " = " << *p << endl;
      // Ensure the parton in question has daughters 
      //       if ( nDa > 0 && ( (p)->daughter(0)->pdgId() == 91 || (p)->daughter(0)->pdgId() == 92 ||
      // 			(p)->daughter(0)->pdgId() == 93) ) {
      if ( nDa > 0 ) {
	if(verbose_) cout << "Has daughters" << endl;
	// Ensure the parton passes some minimum kinematic cuts
	if((p)->pt() > ptMinShower_ && fabs((p)->eta())<etaMaxShower_) {
	  if(verbose_) cout << "Passes kin cuts" << endl;

	  //Main (but simple) workhorse to get all ancestors
	  vector<Candidate const *> allParents;
	  getAncestors( *p, allParents );
	    
	  if(verbose_) {
	    cout << "Parents : " << endl;
	    vector<Candidate const *>::const_iterator iprint = allParents.begin(),
	      iprintend = allParents.end();
	    for ( ; iprint != iprintend; ++iprint ) 
	      cout << **iprint << endl;
	  }
	  
	  // ------------------------------------------------------------
	  // Now identify the flavor and ancestry of the HF Quark
	  // Mother               Origin
	  // ======               =======
	  // incoming quarks      ISR, likely gluon splitting
	  //   light flavor
	  // incoming quarks      ISR, likely flavor excitation
	  //   heavy flavor           
	  // outgoing quark       FSR
	  //   light flavor
	  // outgoing quark       Matrix Element b       
	  //   heavy flavor
	  //     no mother
	  // outgoing quark       Resonance b (e.g. top quark decay)
	  //   heavy flavor
	  //     mother
	  // outgoing resonance   Resonance b (e.g. Higgs decay)
	  // ------------------------------------------------------------
	  vector<Candidate const *>::size_type a_size = allParents.size();
	  int parentIndex=0;

	  // 
	  // Loop over all the ancestors of this parton and find the progenitor.
	  // 
	  bool foundProgenitor = false; 
	  for( vector<Candidate const *>::size_type i=0 ; i < a_size && !foundProgenitor; ++i,++parentIndex ) {
	    const Candidate * aParent=allParents[i];
	    if(verbose_) cout << "Examining parent " << *aParent << endl;
	    vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),aParent);


	    // Get the index of the progenitor candidate
	    progenitorIndex = found - particles.begin();

	    int aParentId = abs(aParent->pdgId());

	    if(verbose_) cout << "Progenitor index = " << progenitorIndex << endl;

	    // Here we examine particles that were produced after the collision
	    if( aParent->numberOfMothers() == 2 && progenitorIndex > 5 ) {
	      // Here is where we have a matrix element
	      if( aParentId == pdgIdToSelect_ ) {
		if(verbose_) cout << "Matrix Element progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_ME;
	      } 
	      // Here we have a gluon splitting from final state radiation
	      else if( (aParentId > 0 && aParentId < FlavorHistory::tQuarkId ) || aParentId==FlavorHistory::gluonId ) {
		if(verbose_) cout << "Gluon splitting progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_GS;
	      }
	      // Here we have a true decay
	      else if( (aParentId>pdgIdToSelect_ && aParentId<FlavorHistory::gluonId) || aParentId > FlavorHistory::gluonId ) {
		if(verbose_) cout << "Flavor decay progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_DECAY;
	      }
	      foundProgenitor = true;
	    }

	    // Here we examine particles that were produced before the collision
	    else if( progenitorIndex==2 || progenitorIndex==3 ) {
	      // Here is a flavor excitation
	      if( aParentId==pdgIdToSelect_ ) {
		if(verbose_) cout << "Flavor excitation progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_EXC;
	      }
	      // Here is gluon splitting from initial state radiation
	      else {		  
		if(verbose_) cout << "Gluon splitting progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_GS;
	      }
	      foundProgenitor = true;
	    }
	  }// End loop over all parents of this parton to find progenitor

	    

	  // 
	  // Now find sister of this particle if there is one
	  // 
	  bool foundSister = false;
	  if ( foundProgenitor ) {
	    // Get the progenitor
	    const Candidate * progenitorCand = particles[progenitorIndex];
	    
	    // Make sure the progenitor has two daughters
	    if ( progenitorCand->numberOfDaughters() == 2 ) {
	      
	      // Get both daughters
	      const Candidate * da1Cand = progenitorCand->daughter(0);
	      const Candidate * da2Cand = progenitorCand->daughter(1);
		
	      // Find which one is NOT this candidate (the other is the sister)
	      const Candidate * sisterCand = 0;
	      if ( da1Cand == particles[partonIndex] ) 
		sisterCand = da2Cand;
	      else
		sisterCand = da1Cand;
	      
	      foundSister = true;
		
	      // Find index of daughter in master list
	      vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),sisterCand);
	      sisterIndex = found - particles.begin();
	      if(verbose_) cout << "Sister index = " << sisterIndex << endl;
	      if ( found != particles.end() )
		if(verbose_) cout << "Sister = " << *found << endl;
	    }
	  }
	  
	}// End if this parton passes some minimal kinematic cuts
      }// End if this particle has strings as daughters
    }// End if this particle was a status==2 parton


    // Add these particles to the flavor history
    flavorHistoryVector.push_back( FlavorHistory( flavorSource, particlesViewH, partonIndex, progenitorIndex, sisterIndex ) ); 
  }


  ValueMap<FlavorHistory>::Filler filler(*flavorHistory);
  filler.insert( particlesViewH, flavorHistoryVector.begin(), flavorHistoryVector.end()  );
  filler.fill();
  // Now add the flavor history to the event record
  evt.put( flavorHistory, flavorHistoryName_ );
}

 
// Helper function to get all ancestors of this candidate
void FlavorHistoryProducer::getAncestors(const Candidate &c,
					 vector<Candidate const *> & moms )
{

  if( c.numberOfMothers() == 1 ) {
    const Candidate * dau = &c;
    const Candidate * mom = c.mother();
    while ( dau->numberOfMothers() != 0) {
      moms.push_back( dau );
      dau = mom ;
      mom = dau->mother();
    } 
  } 
}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( FlavorHistoryProducer );
