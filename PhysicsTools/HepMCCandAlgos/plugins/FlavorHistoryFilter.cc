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
  src_    ( iConfig.getParameter<edm::InputTag>("src" ) ),
  jets_   ( iConfig.getParameter<edm::InputTag>("jets" ) ),
  type_   ( iConfig.getParameter<int> ("type" ) ),
  minPt_  ( iConfig.getParameter<double> ("minPt") ),
  minDR_  ( iConfig.getParameter<double> ("minDR") ),
  scheme_ ( iConfig.getParameter<string> ("scheme") ),
  verbose_( iConfig.getParameter<bool>   ("verbose") )
{
   //now do what ever initialization is needed
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
  
  if ( verbose_) cout << "Looking at GenJetCollection:" << endl;
  for ( ; ijet != ijetEnd; ++ijet ) {
    if ( verbose_) cout << *ijet << endl;
  }

  // boolean to decide to pass this event or not
  bool pass = false;

  // Loop over the objects in the flavor history
  vector<FlavorHistory>::const_iterator i = pFlavorHistory->begin(),
    iend = pFlavorHistory->end();
  for ( ; i != iend && !pass; ++i ) {

    if ( verbose_) cout << "Looking at flavor history object: " << endl;
    if ( verbose_) cout << *i << endl;
    

    // Make sure to consider only flavorSources of "type"
    if ( i->flavorSource() == type_ ) {

      reco::CandidatePtr parton = i->parton();

      // Loop over Genjet collection and find the jet
      // closest the parton in question
      GenJetCollection::const_iterator bestJet = getClosestJet( pJets,
								parton );
      // Check to see if we got any matches
      if ( bestJet != pJets->end() ) {

	if ( verbose_) cout << "Found best jet: " << *bestJet << endl;

	// Have at least one jet within minDR_ of the parton, make decision:
	if ( scheme_ == "deltaR" ) {
	
	  // Here we decide as follows.
	  // If there are no sisters, pass the event.
	  // If there are sisters that are in different jets, pass the event
	  // If there are sisters that are in the same jet, reject the event

	  // No sisters, pass event
	  if ( ! i->hasSister() ) {
	    if ( verbose_) cout << "There is no sister, pass" << endl;
	    pass = true;
	  } 
	  // Has a sister
	  else {
	    if ( verbose_) cout << "Found a sister" << endl;
	    // Find jet closest to sister
	    reco::CandidatePtr sister = i->sister();
	    if ( verbose_) cout << *sister << endl;
	    GenJetCollection::const_iterator sisterJet = getClosestJet( pJets, sister );
	    // Here we found a sister jet
	    if ( sisterJet != pJets->end() ) {
	      if ( verbose_) cout << "sister jet = " << *sisterJet << endl;

	      // If this jet is far enough away from the first jet, pass the event
	      if ( verbose_) cout << "deltaR = " << deltaR( sisterJet->p4(), bestJet->p4() ) << endl;
	      if ( verbose_) cout << "minDR  = " << minDR_ << endl;
	      if ( deltaR( sisterJet->p4(), bestJet->p4() ) > minDR_ ) {
		if ( verbose_) cout << "Pass is true" << endl;
		pass = true;
	      }
	      // Otherwise, fail the event
	      else {
		if ( verbose_) cout << "Pass is false" << endl;
		pass = false;
	      }

	    }// end if has sister jet

	    // Here there is no sister jet, pass event
	    else {
	      if ( verbose_) cout << "No sister jet, pass" << endl;
	      pass = true;
	    }// end if has no sister jet
	  }// end if has sister
	
	}// end if scheme is deltaR
	
	// Otherwise throw exception, at the moment nothing else implemented
	else {
	  throw cms::Exception("FatalError") << "Incorrect scheme for flavor history filter\n"
					     << "Options are: \n" 
					     << "     deltaR\n";
	}
	
      }// End if we found a jet within minDR of the parton
      
    }
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
  double dr = minDR_;
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
