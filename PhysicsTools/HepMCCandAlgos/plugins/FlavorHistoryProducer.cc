// This file was removed but it should not have been.
// This comment is to restore it. 

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "DataFormats/Common/interface/ValueMap.h"
// #include <iterators>


// -------------------------------------------------------------
// Identify the ancestry of the Quark
// 
// 
// Matrix Element:
//    Status 3 parent with precisely 2 "grandparents" that
//    is outside of the "initial" section (0-5) that has the
//    same ID as the status 2 parton in question. 
//    NOTE: This is not the actual ultimate progenitor,
//    but this is the signature of matrix element decays.
//    The ultimate progenitor is the parent of the status 3
//    parton.
//
// Flavor excitation:
//    Almost the same as the matrix element classification,
//    but has only one outgoing parton product instead of two.
//
// Gluon splitting:
//    Parent is a quark of a different flavor than the parton
//    in question, or a gluon. Can come from either ISR or FSR.
//
// True decay:
//    Decays from a resonance like top, Higgs, etc.
// -------------------------------------------------------------

using namespace std;
using namespace reco;
using namespace edm;


ostream & operator<<( ostream & out, Candidate const & cand) 
{
  char buff[1000];
  sprintf(buff, "%5d, status = %5d, nmo = %5d, nda = %5d, pt = %6.2f, eta = %6.2f, phi = %6.2f, m = %6.2f", 
	  cand.pdgId(), cand.status(), 
	  cand.numberOfMothers(),
	  cand.numberOfDaughters(),
	  cand.pt(), cand.eta(), cand.phi(), cand.mass() );
  out << buff;
  return out;
}

ostream & operator<<( ostream & out, FlavorHistory const & cand) 
{
  out << "Source     = " << cand.flavorSource() << endl;
  if ( cand.hasParton() ) 
    out << "Parton     = " << cand.parton().key() << " : " << *(cand.parton()) << endl;
  if ( cand.hasProgenitor() ) 
    out << "Progenitor = " << cand.progenitor().key() << " : " << *(cand.progenitor()) << endl;
  if ( cand.hasSister() ) 
    out << "Sister     = " << cand.sister().key() << " : " << *(cand.sister()) << endl;
  if ( cand.hasParton() ) {
    out << "Ancestry: " << endl;
    Candidate const * ipar = cand.parton()->mother();
    while ( ipar->numberOfMothers() > 0 ) {
      out << *ipar << endl;
      ipar = ipar->mother();
    }
  }
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
  produces<vector<FlavorHistory> >(flavorHistoryName_);
}

FlavorHistoryProducer::~FlavorHistoryProducer() { 
}

void FlavorHistoryProducer::beginJob( const EventSetup & es ) {
}

void FlavorHistoryProducer::produce( Event& evt, const EventSetup& ) 
{

  if ( verbose_ ) cout << "Producing flavor history" << endl;
  
  // Get a handle to the particle collection (OwnVector)
  Handle<CandidateView > particlesViewH;
  evt.getByLabel( src_, particlesViewH );

  // Copy the View to an vector for easier iterator manipulation convenience
  vector<const Candidate* > particles;
  for( CandidateView::const_iterator p = particlesViewH->begin();  p != particlesViewH->end(); ++p ) {
    particles.push_back(&*p);
  }
  
  // Make a new flavor history vector
  auto_ptr<vector<FlavorHistory> > flavorHistoryVector ( new vector<FlavorHistory> () ) ;

  // ------------------------------------------------------------
  // Loop over partons
  // ------------------------------------------------------------
  vector<const Candidate* >::size_type j;
  vector<const Candidate* >::size_type j_max = particles.size();
  for( j=0; j<j_max; ++j ) {

    if ( verbose_ ) cout << "Processing particle " << j << endl;

    // Get the candidate
    const Candidate *p = particles[j];
    // Set up indices that we'll need for the flavor history
    vector<Candidate const *>::size_type partonIndex=j;
    vector<Candidate const *>::size_type progenitorIndex=0;
    vector<Candidate const *>::size_type sisterIndex=0;
    bool foundProgenitor = false; 
    bool foundSister = false;
    FlavorHistory::FLAVOR_T flavorSource=FlavorHistory::FLAVOR_NULL;


    int idabs = abs( (p)->pdgId() );
    int nDa = (p)->numberOfDaughters();

    // Check if we have a status 2 or 3 particle, which is a parton before the string.
    // Only consider quarks. 
    if ( idabs <= FlavorHistory::tQuarkId && p->status() == 2 ) {
      // Ensure the parton in question has daughters 
      if ( nDa > 0 ) {
	// Ensure the parton passes some minimum kinematic cuts
	if((p)->pt() > ptMinShower_ && fabs((p)->eta())<etaMaxShower_) {


 	  if(verbose_) cout << "--------------------------" << endl;
 	  if(verbose_) cout << "Processing particle " << j  << " = " << *p << endl;


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
	  

	  // -------------------------------------------------------------
	  // Identify the ancestry of the Quark
	  // 
	  // 
	  // Matrix Element:
	  //    Status 3 parent with precisely 2 "grandparents" that
	  //    is outside of the "initial" section (0-5) that has the
	  //    same ID as the status 2 parton in question. 
	  //    NOTE: This is not the actual ultimate progenitor,
	  //    but this is the signature of matrix element decays.
	  //    The ultimate progenitor is the parent of the status 3
	  //    parton.
	  //
	  // Flavor excitation:
	  //    Almost the same as the matrix element classification,
	  //    but has only one outgoing parton product instead of two.
	  //
	  // Gluon splitting:
	  //    Parent is a quark of a different flavor than the parton
	  //    in question, or a gluon. Can come from either ISR or FSR.
	  //
	  // True decay:
	  //    Decays from a resonance like top, Higgs, etc.
	  // -------------------------------------------------------------
	  vector<Candidate const *>::size_type a_size = allParents.size();
	  int parentIndex=0;

	  // 
	  // Loop over all the ancestors of this parton and find the progenitor.
	  // 
 	  for( vector<Candidate const *>::size_type i=0 ; i < a_size && !foundProgenitor; ++i,++parentIndex ) {
	    const Candidate * aParent=allParents[i];
	    vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),aParent);


	    // Get the index of the progenitor candidate
	    progenitorIndex = found - particles.begin();

	    int aParentId = abs(aParent->pdgId());

	    // -----------------------------------------------------------------------
	    // Here we examine particles that were produced after the collision
	    // -----------------------------------------------------------------------
 	    if( aParent->numberOfMothers() == 2 && progenitorIndex > 5 ) {
	      // Here is where we have a matrix element process.
	      // The signature is to have a status 3 parent with precisely 2 parents
	      // that is outside of the "initial" section (0-5) that has the
	      // same ID as the status 2 parton in question.
	      // NOTE: This is not the actual ultimate progenitor, but this is
	      // the signature of matrix element decays. The ultimate progentitor
	      // is the parent of the status 3 parton. 
	      // ALSO NOTE: If this parton has no sister, then this will be classified as
	      // a flavor excitation. The only difference, since the initial states are
	      // mostly gluons, is that the matrix element cases have a sister,
	      // while the flavor excitation cases do not. 
	      // If we do not find a sister below, this will be classified as a flavor
	      // excitation. 
	      if( aParentId == pdgIdToSelect_ ) {
		if(verbose_) cout << "Matrix Element progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_ME;

		// The "true" progenitor is the next parent in the list (the parent of this
		// progenitor).
		if ( i != a_size - 1 ) {
		  const Candidate * progenitorCand = allParents[i+1];
		  vector<Candidate const *>::const_iterator foundAgain = find( particles.begin(),
									       particles.end(),
									       progenitorCand );
		  progenitorIndex = foundAgain - particles.begin();
		  foundProgenitor = true;

		} else {
		  edm::LogWarning("UnexpectedFormat") << "Error! Parentage information in FlavorHistoryProducer is not what is expected";
		  
		  cout << "Particle : " << *p << endl;
		  cout << "Parents : " << endl;
		  vector<Candidate const *>::const_iterator iprint = allParents.begin(),
		    iprintend = allParents.end();
		  for ( ; iprint != iprintend; ++iprint ) 
		    cout << **iprint << endl;

		  foundProgenitor = false;

		}
	      } 
	      // Here we have a gluon splitting from final state radiation. 
	      // The parent is a quark of a different flavor, or a gluon, in the
	      // final state. 
	      else if( (aParentId > 0 && aParentId < FlavorHistory::tQuarkId ) || aParentId==FlavorHistory::gluonId ) {
		if(verbose_) cout << "Gluon splitting progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_GS;
		foundProgenitor = true;
	      }
	      // Here we have a true decay. Parent is not a quark or a gluon.
	      else if( (aParentId>pdgIdToSelect_ && aParentId<FlavorHistory::gluonId) || aParentId > FlavorHistory::gluonId ) {
		if(verbose_) cout << "Flavor decay progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_DECAY;
		foundProgenitor = true;
	      }
	    }

	    // -----------------------------------------------------------------------
	    // Here we examine particles that were produced before the collision
	    // -----------------------------------------------------------------------
	    else if( progenitorIndex <= 5 ) {
	      // Parent has a quark daughter equal and opposite to this: ISR
	      if( aParent->numberOfDaughters() > 0 && 
		  aParent->daughter(0)->pdgId() == -1 * p->pdgId()  ) {
		if(verbose_) cout << "Gluon splitting progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_GS;
	      }
	      // Otherwise, this is flavor excitation. Rarely happens because
	      // mostly the initial state is gluons which will be caught by the
	      // "matrix element" version above. 
	      else {		  
		if(verbose_) cout << "Flavor excitation progenitor" << endl;
		flavorSource = FlavorHistory::FLAVOR_EXC;
	      }
	      foundProgenitor = true;
	    }
	  }// End loop over all parents of this parton to find progenitor

	    

	  // 
	  // Now find sister of this particle if there is one
	  // 
	  if ( foundProgenitor && progenitorIndex >= 0 ) {
	    // Get the progenitor
	    const Candidate * progenitorCand = particles[progenitorIndex];

	    if ( verbose_ ) cout << "Found progenitor: " << *progenitorCand << endl;

	    // Here is the normal case of a sister around
	    if ( progenitorCand->numberOfDaughters() >= 2 ) {
	      const Candidate * sisterCand = 0;
	      
	      for ( unsigned int iida = 0; iida < progenitorCand->numberOfDaughters(); ++iida ) {
		const Candidate * dai = progenitorCand->daughter(iida);

		if ( verbose_ ) cout << "Sister candidate " << *dai << endl;
		
		if ( dai->pdgId() == -1 * p->pdgId() ) {
		  if ( verbose_ ) cout << "Found actual sister" << endl;
		  sisterCand = dai;
		  foundSister = true;
		}
	      }
		
	      if ( foundSister ) {
		// Find index of daughter in master list
		vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),sisterCand);
		sisterIndex = found - particles.begin();
		if(verbose_) cout << "Sister index = " << sisterIndex << endl;
		if ( found != particles.end() )
		  if(verbose_) cout << "Sister = " << **found << endl;
	      } // end if found sister
	    }
	    // Here is if we have a "transient" decay in the code that isn't
	    // really a decay, so we need to look at the parent of the progenitor
	    else {
	      const Candidate * grandProgenitorCand = progenitorCand->mother(0);
	      const Candidate * sisterCand = 0;

	      if ( verbose_ ) cout << "Looking for sister, progenitor is " << *progenitorCand << endl;
	    
	      // Make sure the progenitor has two daughters
	      if ( grandProgenitorCand->numberOfDaughters() >= 2 ) {

		for ( unsigned int iida = 0; iida < grandProgenitorCand->numberOfDaughters(); ++iida ) {
		  const Candidate * dai = grandProgenitorCand->daughter(iida);

		  if ( verbose_ ) cout << "Looking for sister " << *dai << endl;
		
		  if ( dai->pdgId() == -1 * p->pdgId() ) {
		    if ( verbose_ ) cout << "Found sister" << endl;
		    sisterCand = dai;
		    foundSister = true;
		  }
		}
		
		if ( foundSister ) {
		  // Find index of daughter in master list
		  vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),sisterCand);
		  sisterIndex = found - particles.begin();
		  if(verbose_) cout << "Sister index = " << sisterIndex << endl;
		  if ( found != particles.end() )
		    if(verbose_) cout << "Sister = " << **found << endl;
		} // end if found sister
	      } // End of have at least 2 grand progenitor daughters
	    } // End if we have to look at parents of progenitor to find sister

	  } // end if found progenitor

	  // ------
	  // Here, we change the type from matrix element to flavor excitation
	  // if there are no sisters present. 
	  // ------
	  if ( flavorSource == FlavorHistory::FLAVOR_ME && !foundSister ) {
	    flavorSource = FlavorHistory::FLAVOR_EXC;
	  }
	  
	}// End if this parton passes some minimal kinematic cuts
      }// End if this particle has strings as daughters
    }// End if this particle was a status==2 parton

    // Make sure we've actually found a sister and a progenitor
    if ( !foundProgenitor ) progenitorIndex = 0;
    if ( !foundSister ) sisterIndex = 0;

    // We've found the particle, add to the list (status 2 only)
    if ( idabs == pdgIdToSelect_ && p->status() == 2 ) 
      flavorHistoryVector->push_back( FlavorHistory( flavorSource, particlesViewH, partonIndex, progenitorIndex, sisterIndex ) ); 
  }


//   ValueMap<FlavorHistory>::Filler filler(*flavorHistory);
//   filler.insert( particlesViewH, flavorHistoryVector.begin(), flavorHistoryVector.end()  );
//   filler.fill();
  // Now add the flavor history to the event record
  if ( verbose_ ) {
    cout << "Outputting pdg id = " << pdgIdToSelect_ << " with nelements = " << flavorHistoryVector->size() << endl;
    vector<FlavorHistory>::const_iterator i = flavorHistoryVector->begin(),
      iend = flavorHistoryVector->end();
    for ( ; i !=iend; ++i ) {
      cout << *i << endl;
    }
  }
  evt.put( flavorHistoryVector, flavorHistoryName_ );
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
