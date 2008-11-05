// This file was removed but it should not have been.
// This comment is to restore it. 

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryFilter.h"
#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"

#include <vector>

using namespace edm;
using namespace reco;
using namespace std;


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FlavorHistoryFilter::FlavorHistoryFilter(const edm::ParameterSet& iConfig) :
  src_           ( iConfig.getParameter<edm::InputTag>("src" ) ),
  jets_          ( iConfig.getParameter<edm::InputTag>("jets" ) ),
  type_          ( iConfig.getParameter<int> ("type" ) ),
  matchDR_       ( iConfig.getParameter<double> ("matchDR") ),
  minPt_         ( iConfig.getParameter<double> ("minPt") ),
  minDR_         ( iConfig.getParameter<double> ("minDR") ),
  maxDR_         ( iConfig.getParameter<double> ("maxDR") ),
  scheme_        ( iConfig.getParameter<string> ("scheme") ),
  requireSisters_( iConfig.getParameter<bool>   ("requireSisters") ),
  verbose_       ( iConfig.getParameter<bool>   ("verbose") )
{

}


FlavorHistoryFilter::~FlavorHistoryFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//---------------------------------------------------------------------------
//   FlavorHistoryFilter
//   This will filter events as follows:
//   - Inputs vector<FlavorHistory> from FlavorHistoryProducer
//   - Inputs GenJetCollection
//   - If there are no FlavorHistory's that have flavorSource of "type",
//     then the event is rejected.
//   - If there is at least one FlavorHistory that has flavorSource of "type",
//     then we examine the kinematic criteria:
//        - For delta R method, if there is a sister of the parton
//          that is within "minDR" of "this" parton, the event is rejected,
//          otherwise it is passed.
//        - For the pt method, if the parton itself is less than a pt
//          threshold, it is rejected, and if it is above, it is passed
//---------------------------------------------------------------------------
bool
FlavorHistoryFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the flavor history
  Handle<vector<FlavorHistory> > pFlavorHistory;
  iEvent.getByLabel(src_,pFlavorHistory);

  // Get the jet collection
  Handle<GenJetCollection> pJets;
  iEvent.getByLabel(jets_, pJets );

  GenJetCollection::const_iterator ijetBegin = pJets->begin(),
    ijetEnd = pJets->end(),
    ijet = ijetBegin;
  
//   if ( verbose_) cout << "Looking at GenJetCollection:" << endl;
//   for ( ; ijet != ijetEnd; ++ijet ) {
//     if ( verbose_) cout << *ijet << endl;
//   }

  // boolean to decide to pass this event or not
  bool pass = false;
  
  // Loop over the objects in the flavor history
  vector<FlavorHistory>::const_iterator i = pFlavorHistory->begin(),
    iend = pFlavorHistory->end();
  for ( ; i != iend; ++i ) {

    if ( verbose_) cout << "Looking at flavor history object: " << endl;
    if ( verbose_) cout << *i << endl;



    reco::CandidatePtr parton = i->parton();

    // Loop over Genjet collection and find the jet
    // closest the parton in question
    GenJetCollection::const_iterator bestJet = getClosestJet( pJets,
							      parton ),
      jetsEnd = pJets->end();

    // Check to see if we got any genjet matches
    if ( bestJet != jetsEnd ) {


      if ( verbose_ ) cout << "Found best jet: " << *bestJet << endl;
      if ( verbose_ ) cout << "delta R to parton = " << deltaR( bestJet->p4(), parton->p4() ) << endl;

      // Make sure to consider only flavorSources of "type".
      // If "type < 0" then we ignore this. 
      if ( i->flavorSource() == type_ || type_ < 0 ) {

	if ( verbose_ ) cout << "Flavor source matches" << endl;
	
	
	// Have at least one jet within minDR_ of the parton, make decision:
	if ( scheme_ == "deltaR" ) {
	  
	  // Here we decide as follows.
	  // If there are sisters that satisfy minDr < dr < maxDr, pass the event
	  // If there are sisters that satisfy dr < minDr || dr > maxDr, reject the event
	  // If there are no sisters: 
	  //      If we require sisters, then we fail the event. 
	  //      Otherwise, we pass. 
	  
	  // No sisters
	  if ( !i->hasSister() ) {
	    if ( requireSisters_ ) 
	      pass |= false;
	    else 
	      pass |= true;
	  } 
	  // Has a sister parton
	  else {
	    if ( verbose_) cout << "Found a sister" << endl;
	    // Find jet closest to sister
	    reco::CandidatePtr sister = i->sister();
	    if ( verbose_) cout << *sister << endl;
	    GenJetCollection::const_iterator sisterJet = getClosestJet( pJets, sister );

	    // Here we found a sister jet
	    if ( sisterJet != jetsEnd && sisterJet != bestJet ) {
	      if ( verbose_) cout << "sister jet = " << *sisterJet << endl;

	      // If this jet is far enough away from the first jet, pass the event
	      if ( verbose_) cout << "deltaR = " << deltaR( sisterJet->p4(), bestJet->p4() ) << endl;
	      if ( verbose_) cout << "minDR  = " << minDR_ << endl;
	      double dr = deltaR( sisterJet->p4(), bestJet->p4() );
	      if (  dr > minDR_ &&
		    dr < maxDR_ ) {
		pass |= true;
	      }
	      // Otherwise, fail the event
	      else {
		pass |= false;
	      }

	    }// end if has sister jet

	    // Here there is no sister jet
	    else {
	      if ( requireSisters_ ) {
		pass |= false;
	      }
	      else {
		pass |= true;
	      }
	    }// end if has no sister jet
	  }// end if has sister parton
	
	}// end if scheme is deltaR
	
	// Otherwise throw exception, at the moment nothing else implemented
	else {
	  throw cms::Exception("FatalError") << "Incorrect scheme for flavor history filter\n"
					     << "Options are: \n" 
					     << "     deltaR\n";
	}
	
      }// end if flavor source is the type we're looking for  
    }// End if we found a genjet within minDR of the parton
  }

  if ( verbose_) cout << "About to return pass = " << pass << endl;
  return pass;

}

// ------------ method called once each job just before starting event loop  ------------
void 
FlavorHistoryFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
FlavorHistoryFilter::endJob() {
}


GenJetCollection::const_iterator 
FlavorHistoryFilter::getClosestJet( Handle<GenJetCollection> const & pJets,
				    reco::CandidatePtr const & parton ) const 
{
  double dr = matchDR_;
  GenJetCollection::const_iterator j = pJets->begin(),
    jend = pJets->end();
  GenJetCollection::const_iterator bestJet = pJets->end();
  for ( ; j != jend; ++j ) {
    double dri = deltaR( parton->p4(), j->p4() );
    if ( dri < dr ) {
      dr = dri;
      bestJet = j;
    }
  }
  return bestJet;
}

//define this as a plug-in
DEFINE_FWK_MODULE(FlavorHistoryFilter);
