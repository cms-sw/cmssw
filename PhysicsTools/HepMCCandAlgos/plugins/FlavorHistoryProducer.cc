// This file was removed but it should not have been.
// This comment is to restore it. 

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

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
//
// Flavor excitation:
//    If we find only one outgoing parton.
//
// Gluon splitting:
//    Parent is a quark of a different flavor than the parton
//    in question, or a gluon. 
//    Can come from either ISR or FSR.
//
// True decay:
//    Decays from a resonance like top, Higgs, etc.
// -------------------------------------------------------------

using namespace std;
using namespace reco;
using namespace edm;

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

void FlavorHistoryProducer::produce( Event& evt, const EventSetup& ) 
{
  if ( verbose_ ) cout << "---------- Hello from FlavorHistoryProducer! -----" << endl;
  
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


    int idabs = std::abs( (p)->pdgId() );
    int nDa = (p)->numberOfDaughters();

    // Check if we have a status 2 or 3 particle, which is a parton before the string.
    // Only consider quarks. 
    if ( idabs <= FlavorHistory::tQuarkId && p->status() == 2 &&
	 idabs == pdgIdToSelect_ ) {
      // Ensure the parton in question has daughters 
      if ( nDa > 0 ) {
	// Ensure the parton passes some minimum kinematic cuts
	if((p)->pt() > ptMinShower_ && fabs((p)->eta())<etaMaxShower_) {

 	  if(verbose_) cout << "--------------------------" << endl;
 	  if(verbose_) cout << "Processing particle " << j << ", status = " << p->status() << ", pdg id = " << p->pdgId() << endl;


	  //Main (but simple) workhorse to get all ancestors
	  vector<Candidate const *> allParents;
	  getAncestors( *p, allParents );
	    
	  if(verbose_) {
	    cout << "Parents : " << endl;
	    vector<Candidate const *>::const_iterator iprint = allParents.begin(),
	      iprintend = allParents.end();
	    for ( ; iprint != iprintend; ++iprint ) 
	      cout << " status = " << (*iprint)->status() << ", pdg id = " << (*iprint)->pdgId() << ", pt = " << (*iprint)->pt() << endl;
	  }
	  
	  // -------------------------------------------------------------
	  // Identify the ancestry of the Quark
	  // 
	  // 
	  // Matrix Element:
	  //    Status 3 parent with precisely 2 "grandparents" that
	  //    is outside of the "initial" section (0-5) that has the
	  //    same ID as the status 2 parton in question. 
	  //
	  // Flavor excitation:
	  //    If we find only one outgoing parton.
	  //
	  // Gluon splitting:
	  //    Parent is a quark of a different flavor than the parton
	  //    in question, or a gluon. 
	  //    Can come from either ISR or FSR.
	  //
	  // True decay:
	  //    Decays from a resonance like top, Higgs, etc.
	  // -------------------------------------------------------------
	  vector<Candidate const *>::size_type a_size = allParents.size();

	  // 
	  // Loop over all the ancestors of this parton and find the progenitor.
	  // 
 	  for( vector<Candidate const *>::size_type i=0 ; i < a_size && !foundProgenitor; ++i ) {
	    const Candidate * aParent=allParents[i];
	    vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),aParent);


	    // Get the index of the progenitor candidate
	    progenitorIndex = found - particles.begin();

	    int aParentId = std::abs(aParent->pdgId());
	    
	    // -----------------------------------------------------------------------
	    // Here we examine particles that were produced after the collision
	    // -----------------------------------------------------------------------
 	    if( progenitorIndex > 5 ) {
	      // Here is where we have a matrix element process.
	      // The signature is to have a status 3 parent with precisely 2 parents
	      // that is outside of the "initial" section (0-5) that has the
	      // same ID as the status 2 parton in question.
	      // NOTE: If this parton has no sister, then this will be classified as
	      // a flavor excitation. Often the only difference is that the matrix element 
	      // cases have a sister, while the flavor excitation cases do not. 
	      // If we do not find a sister below, this will be classified as a flavor
	      // excitation. 
	      if(  aParent->numberOfMothers() == 2 &&  
		  aParent->pdgId() == p->pdgId() && aParent->status() == 3 ) {
		if(verbose_) cout << "Matrix Element progenitor" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_ME;
		foundProgenitor = true;
	      } 
	      // Here we have a true decay. Parent is not a quark or a gluon.
	      else if( (aParentId>pdgIdToSelect_ && aParentId<FlavorHistory::gluonId) || 
		       aParentId > FlavorHistory::gluonId ) {
		if(verbose_) cout << "Flavor decay progenitor" << endl;
		if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_DECAY;
		foundProgenitor = true;
	      }
	    }// end if progenitorIndex > 5
	  }// end loop over parents

	  // Otherwise, classify it as gluon splitting. Take the first status 3 parton with 1 parent
	  // that you see as the progenitor
	  if ( !foundProgenitor  ) {
	    if ( flavorSource == FlavorHistory::FLAVOR_NULL ) flavorSource = FlavorHistory::FLAVOR_GS;
	    // Now get the parton with only one parent (the proton) and that is the progenitor
	    for( vector<Candidate const *>::size_type i=0 ; i < a_size && !foundProgenitor; ++i ) {
	      const Candidate * aParent=allParents[i];
	      vector<Candidate const *>::const_iterator found = find(particles.begin(),particles.end(),aParent);
	      // Get the index of the progenitor candidate
	      progenitorIndex = found - particles.begin();
	      
	      if ( aParent->numberOfMothers() == 1 && aParent->status() == 3 ) {
		foundProgenitor = true;
	      }
	    }// end loop over parents
	  }// end if we haven't found a progenitor, and if we haven't found a status 3 parton of the same flavor
	   // (i.e. end if examining gluon splitting)
	}// End if this parton passes some minimal kinematic cuts
      }// End if this particle is status 2 (has strings as daughters)



      // Make sure we've actually found a progenitor
      if ( !foundProgenitor ) progenitorIndex = j_max;

      // We've found the particle, add to the list

      partonIndices.push_back( partonIndex );
      progenitorIndices.push_back( progenitorIndex );
      flavorSources.push_back(flavorSource);
      sisterIndices.push_back( -1 ); // set below
	
    }// End if this particle was a status==2 parton
  }// end loop over particles

  // Now add sisters.
  // Also if the event is preliminarily classified as "matrix element", check to
  // make sure that they have a sister. If not, it is flavor excitation. 
  
  if ( verbose_ ) cout << "Making sisters" << endl;
  // First make sure nothing went terribly wrong:
  if ( partonIndices.size() == progenitorIndices.size() && partonIndices.size() > 0 ) {
    // Now loop over the candidates
    for ( unsigned int ii = 0; ii < partonIndices.size(); ++ii ) {
      // Get the iith particle
      const Candidate * candi = particles[partonIndices[ii]];
      if ( verbose_ ) cout << "   Sister candidate particle 1:  " << ii << ", pdgid = " << candi->pdgId() << ", status = " << candi->status() << endl;
      // Get the iith progenitor
      // Now loop over the other flavor history candidates and
      // attempt to find a sister to this one
      for ( unsigned int jj = 0; jj < partonIndices.size(); ++jj ) {
	if ( ii != jj ) {
	  const Candidate * candj = particles[partonIndices[jj]];
      if ( verbose_ ) cout << "   Sister candidate particle 2:  " << jj << ", pdgid = " << candj->pdgId() << ", status = " << candj->status() << endl;
	  // They should be opposite in pdgid and have the same status, and
	  // the same progenitory.
	  if ( candi->pdgId() == -1 * candj->pdgId() && candi->status() == candj->status()  ) {
	    sisterIndices[ii] = partonIndices[jj];
	    if ( verbose_ ) cout << "Parton " << partonIndices[ii] << " has sister " << sisterIndices[ii] << endl;
	  }
	}
      }

      // Here, ensure that there is a sister. Otherwise this is "flavor excitation"
      if ( sisterIndices[ii] < 0 ) {
	if ( verbose_ ) cout << "No sister. Classified as flavor excitation" << endl;
	flavorSources[ii] = FlavorHistory::FLAVOR_EXC;
      } 

      if ( verbose_ ) cout << "Getting matched jet" << endl;
      // Get the closest match in the matched object collection
      CandidateView::const_iterator matched = getClosestJet( matchedView, *candi );
      CandidateView::const_iterator matchedBegin = matchedView->begin();
      CandidateView::const_iterator matchedEnd = matchedView->end();

      
      if ( verbose_ ) cout << "Getting sister jet" << endl;
      // Get the closest sister in the matched object collection, if sister is found
      CandidateView::const_iterator sister = 
	( sisterIndices[ii] >= 0 && static_cast<unsigned int>(sisterIndices[ii]) < particles.size() ) ? 
	getClosestJet( matchedView, *particles[sisterIndices[ii]] ) :
	matchedEnd ;

      if ( verbose_ ) cout << "Making matched shallow clones : ";
      ShallowClonePtrCandidate matchedCand ;
      if ( matched != matchedEnd ) {
	if ( verbose_ ) cout << " found" << endl;
	matchedCand = ShallowClonePtrCandidate( CandidatePtr(matchedView, matched - matchedBegin ) );
      } else {
	if ( verbose_ ) cout << " NOT found" << endl;
      }
      
      if ( verbose_ ) cout << "Making sister shallow clones : ";
      ShallowClonePtrCandidate sisterCand;
      if ( sister != matchedEnd ) {
	if ( verbose_ ) cout << " found" << endl;
	sisterCand = ShallowClonePtrCandidate( CandidatePtr(matchedView, sister - matchedBegin ) );
      } else {
	if ( verbose_ ) cout << " NOT found" << endl;
      }
      
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
  
  // If we got any partons, cache them and then write them out
  if ( flavorHistoryEvent->size() > 0 ) {

    // Calculate some nice variables for FlavorHistoryEvent
    if ( verbose_ ) cout << "Caching flavor history event" << endl;
    flavorHistoryEvent->cache();

    if ( verbose_ ) {
      cout << "Outputting pdg id = " << pdgIdToSelect_ << " with nelements = " << flavorHistoryEvent->size() << endl;
      vector<FlavorHistory>::const_iterator i = flavorHistoryEvent->begin(),
	iend = flavorHistoryEvent->end();
      for ( ; i !=iend; ++i ) {
	cout << *i << endl;
      }
    }
  }

  // Now add the flavor history to the event record
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
