// This file was removed but it should not have been.
// This comment is to restore it. 

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"

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
  matchedSrc_( p.getParameter<InputTag>( "matchedSrc") ),
  matchDR_ ( p.getParameter<double> ("matchDR") ),
  pdgIdToSelect_( p.getParameter<int> ("pdgIdToSelect") ),
  ptMinParticle_( p.getParameter<double>( "ptMinParticle") ),  
  ptMinShower_( p.getParameter<double>( "ptMinShower") ),  
  etaMaxParticle_( p.getParameter<double>( "etaMaxParticle" )),  
  etaMaxShower_( p.getParameter<double>( "etaMaxShower" )),
  flavorHistoryName_( p.getParameter<string>("flavorHistoryName") ),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) )
{
  produces<FlavorHistoryEvent >(flavorHistoryName_);
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

  Handle<CandidateView> matchedView;
  evt.getByLabel( matchedSrc_, matchedView );

  // Copy the View to an vector for easier iterator manipulation convenience
  vector<const Candidate* > particles;
  for( CandidateView::const_iterator p = particlesViewH->begin();  p != particlesViewH->end(); ++p ) {
    particles.push_back(&*p);
  }

  // List of indices for the partons to add
  vector<int> partonIndices;
  // List of progenitors for those partons
  vector<int> progenitorIndices;
  // List of sisters for those partons
  vector<int> sisterIndices;
  // Flavor sources
  vector<FlavorHistory::FLAVOR_T> flavorSources;
  
  // Make a new flavor history vector
  auto_ptr<FlavorHistoryEvent > flavorHistoryEvent ( new FlavorHistoryEvent () ) ;

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
    vector<Candidate const *>::size_type progenitorIndex=j_max;
    bool foundProgenitor = false; 
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
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_ME;

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
	      // Here we have a true decay. Parent is not a quark or a gluon.
	      else if( (aParentId>pdgIdToSelect_ && aParentId<FlavorHistory::gluonId) || aParentId > FlavorHistory::gluonId ) {
		if(verbose_) cout << "Flavor decay progenitor" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_DECAY;
		foundProgenitor = true;
	      }
	      // Here we have a gluon splitting from final state radiation. 
	      // The parent is a quark of a different flavor, or a gluon, in the
	      // final state. 
	      // NOTE! It is possible that this is actually a gluon splitting event. This will
	      // be checked at the end when examining the outgoing parton's sisters. 
	      else if( (aParentId > 0 && aParentId < FlavorHistory::tQuarkId ) || aParentId==FlavorHistory::gluonId ) {
		if(verbose_) cout << "Gluon splitting progenitor from FSR" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_GS;
		foundProgenitor = true;
	      }
	    }

	    // -----------------------------------------------------------------------
	    // Here we examine particles that were produced before the collision
	    // -----------------------------------------------------------------------
	    else if( progenitorIndex <= 5 ) {
	      // Parent is a gluon from before the interaction: ISR. 
	      // NOTE! It is possible that this is actually a gluon splitting event. This will
	      // be checked at the end when examining the outgoing parton's sisters. 
	      if( aParent->numberOfDaughters() > 0 && 
		  aParent->daughter(0)->pdgId() == -1 * p->pdgId()  ) {
		if(verbose_) cout << "Gluon splitting progenitor from ISR" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_GS;
	      }
	      // Otherwise, this is flavor excitation. Rarely happens because
	      // mostly the initial state is gluons which will be caught by the
	      // "matrix element" version above. 
	      else {		  
		if(verbose_) cout << "Flavor excitation progenitor" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_EXC;
	      }
	      foundProgenitor = true;
	    }
	  }// End loop over all parents of this parton to find progenitor
	}// End if this parton passes some minimal kinematic cuts
      }// End if this particle has strings as daughters
    }// End if this particle was a status==2 parton


    // Make sure we've actually found a progenitor
    if ( !foundProgenitor ) progenitorIndex = j_max;

    // We've found the particle, add to the list (status 2 only)
    if ( idabs == pdgIdToSelect_ && p->status() == 2 ) {
      partonIndices.push_back( partonIndex );
      progenitorIndices.push_back( progenitorIndex );
      flavorSources.push_back(flavorSource);
      sisterIndices.push_back( -1 ); // set below
    }
  }// end loop over particles

  // Now add sisters.
  // Also if the event is preliminarily classified as "matrix element", check to
  // make sure that they have a sister. If not, it is flavor excitation. 
  
  if ( verbose_ ) cout << "Making sisters" << endl;
  // First make sure nothing went terribly wrong:
  if ( partonIndices.size() == progenitorIndices.size() ) {
    // Now loop over the candidates
    for ( unsigned int ii = 0; ii < partonIndices.size(); ++ii ) {
      // Get the iith particle
      const Candidate * candi = particles[partonIndices[ii]];
      // Get the iith progenitor
      // Now loop over the other flavor history candidates and
      // attempt to find a sister to this one
      for ( unsigned int jj = 0; jj < partonIndices.size(); ++jj ) {
	if ( ii != jj ) {
	  const Candidate * candj = particles[partonIndices[jj]];
	  // They should be opposite in pdgid and have the same status, and
	  // the same progenitory.
	  if ( candi->pdgId() == -1 * candj->pdgId() && candi->status() == candj->status() 
	       && progenitorIndices[ii] == progenitorIndices[jj] ) {
	    sisterIndices[ii] = partonIndices[jj];
	    if ( verbose_ ) cout << "Parton " << partonIndices[ii] << " has sister " << sisterIndices[ii] << endl;
	  }
	}
      }

      // Here, ensure that there is a sister. Otherwise this is "flavor excitation"
      if ( sisterIndices[ii] < 0 ) {
	flavorSources[ii] = FlavorHistory::FLAVOR_EXC;
      }

      if ( verbose_ ) cout << "Getting matched jet" << endl;
      // Get the closest match in the matched object collection
      CandidateView::const_iterator matched = getClosestJet( matchedView, *candi );
      CandidateView::const_iterator matchedBegin = matchedView->begin();
      CandidateView::const_iterator matchedEnd = matchedView->end();

      if ( matched != matchedEnd) 
	if ( verbose_ ) cout << "Matched jet = " << *matched << endl;
      
      if ( verbose_ ) cout << "Getting sister jet" << endl;
      // Get the closest sister in the matched object collection, if sister is found
      CandidateView::const_iterator sister = 
	( sisterIndices[ii] >= 0 && static_cast<unsigned int>(sisterIndices[ii]) < particles.size() ) ? 
	getClosestJet( matchedView, *particles[sisterIndices[ii]] ) :
	matchedEnd ;

      if ( sister != matchedEnd ) 
	if ( verbose_ ) cout << "Sister jet = " << *sister << endl;

      if ( verbose_ ) cout << "Making matched shallow clones" << endl;
      ShallowClonePtrCandidate matchedCand ;
      if ( matched != matchedEnd ) 
	matchedCand = ShallowClonePtrCandidate( CandidatePtr(matchedView, matched - matchedBegin ) );
      
      
      if ( verbose_ ) cout << "Making sister shallow clones" << endl;
      ShallowClonePtrCandidate sisterCand;
      if ( sister != matchedEnd )  
	sisterCand = ShallowClonePtrCandidate( CandidatePtr(matchedView, sister - matchedBegin ) );
      
      if ( verbose_ ) cout << "Making history object" << endl;
      // Now create the flavor history object
      FlavorHistory history (flavorSources[ii], 
			     particlesViewH, 
			     partonIndices[ii], progenitorIndices[ii], sisterIndices[ii],
			     matchedCand,
			     sisterCand );
      if ( verbose_ ) cout << "Adding flavor history : " << history << endl;
      flavorHistoryEvent->push_back( history ); 
    }
  }
  

  // Calculate some nice variables for FlavorHistoryEvent
  flavorHistoryEvent->cache();

  // Now add the flavor history to the event record
  if ( verbose_ ) {
    cout << "Outputting pdg id = " << pdgIdToSelect_ << " with nelements = " << flavorHistoryEvent->size() << endl;
    vector<FlavorHistory>::const_iterator i = flavorHistoryEvent->begin(),
      iend = flavorHistoryEvent->end();
    for ( ; i !=iend; ++i ) {
      cout << *i << endl;
    }
  }
  evt.put( flavorHistoryEvent, flavorHistoryName_ );
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


CandidateView::const_iterator 
FlavorHistoryProducer::getClosestJet( Handle<CandidateView> const & pJets,
				      reco::Candidate const & parton ) const 
{
  double dr = matchDR_;
  CandidateView::const_iterator j = pJets->begin(),
    jend = pJets->end();
  CandidateView::const_iterator bestJet = pJets->end();
  for ( ; j != jend; ++j ) {
    double dri = deltaR( parton.p4(), j->p4() );
    if ( dri < dr ) {
      dr = dri;
      bestJet = j;
    }
  }
  return bestJet;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( FlavorHistoryProducer );
