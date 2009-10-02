// This file was removed but it should not have been.
// This comment is to restore it. 

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryFilter.h"
#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistoryProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistorySelectorUtil.h"

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
  bsrc_           ( iConfig.getParameter<edm::InputTag>("bsrc" ) ),
  csrc_           ( iConfig.getParameter<edm::InputTag>("csrc" ) )
{
  if ( iConfig.exists("pathToSelect") )
    pathToSelect_ =        iConfig.getParameter<int>    ("pathToSelect");
  else
    pathToSelect_ = -1;

  // This is the "interface" delta R with which to decide
  // where to take the event from
  dr_                =  iConfig.getParameter<double> ("dr" ) ;
  bool verbose        ( iConfig.getParameter<bool>   ("verbose") );

  // Set up the boundaries.
  // dr0 = 0.0
  // dr1 = set by user
  // dr2 = infinity
  double dr0 = 0.0;
  double dr1 = dr_;
  double dr2 = 99999.0;
  

  // These are the processes that can come from the matrix element calculation
  std::vector<int> me_ids;
  me_ids.push_back(2);   // flavor excitation
  me_ids.push_back(3);   // flavor creation
  
  // These are the processes that can come from the parton shower calculation
  std::vector<int> ps_ids;
  ps_ids.push_back(1);   // gluon splitting


  // To select bb->2 events from matrix element... Path 1 
  bb_me_ = new FlavorHistorySelectorUtil( 5,
					  2,
					  me_ids,
					  dr1,
					  dr2,
					  verbose );

  // To select  b->1 events from matrix element... Path 2 
  b_me_  = new FlavorHistorySelectorUtil( 5,
					  1,
					  me_ids,
					  dr0,
					  dr0,
					  verbose );


  // To select cc->2 events from matrix element... Path 3
  cc_me_ = new FlavorHistorySelectorUtil( 4,
					  2,
					  me_ids,
					  dr1,
					  dr2,
					  verbose );

  // To select  c->1 events from matrix element... Path 4
  c_me_  = new FlavorHistorySelectorUtil( 4,
					  1,
					  me_ids,
					  dr0,
					  dr0,
					  verbose );

  // To select bb->2 events from parton shower ... Path 5 
  b_ps_  = new FlavorHistorySelectorUtil( 5,
					  1,
					  ps_ids,
					  dr0,
					  dr1,
					  verbose );

  
  // To select cc->2 events from parton shower ... Path 6 
  c_ps_  = new FlavorHistorySelectorUtil( 4,
					  1,
					  ps_ids,
					  dr0,
					  dr1,
					  verbose );  

  // To select bb->1 events from matrix element... Path 7
  bb_me_comp_ = new FlavorHistorySelectorUtil( 5,
					       2,
					       me_ids,
					       dr0,
					       dr1,
					       verbose );

  // To select cc->1 events from matrix element... Path 8 
  cc_me_comp_ = new FlavorHistorySelectorUtil( 4,
					       2,
					       me_ids,
					       dr0,
					       dr1,
					       verbose );

  // To select bb->2 events from parton shower ... Path 9 
  b_ps_comp_  = new FlavorHistorySelectorUtil( 5,
					       2,
					       ps_ids,
					       dr1,
					       dr2,
					       verbose );

  // To select cc->1 events from parton shower ... Path 10
  c_ps_comp_  = new FlavorHistorySelectorUtil( 4,
					       2,
					       ps_ids,
					       dr1,
					       dr2,
					       verbose );

  // The veto of all of these is               ... Path 11
  

  // This will write 1-11 (the path number), or 0 if error. 
  produces<unsigned int>();
}


FlavorHistoryFilter::~FlavorHistoryFilter()
{
 
  if ( bb_me_ ) delete bb_me_; 
  if (  b_me_ ) delete b_me_;
  if ( cc_me_ ) delete cc_me_; 
  if (  c_me_ ) delete c_me_;  
  if (  b_ps_ ) delete b_ps_; 
  if (  c_ps_ ) delete c_ps_; 

  if ( bb_me_comp_ ) delete bb_me_comp_; 
  if ( cc_me_comp_ ) delete cc_me_comp_; 
  if (  b_ps_comp_ ) delete b_ps_comp_; 
  if (  c_ps_comp_ ) delete c_ps_comp_;



 
}



bool
FlavorHistoryFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the flavor history
  Handle<FlavorHistoryEvent > bFlavorHistoryEvent;
  iEvent.getByLabel(bsrc_,bFlavorHistoryEvent);

  Handle<FlavorHistoryEvent > cFlavorHistoryEvent;
  iEvent.getByLabel(csrc_,cFlavorHistoryEvent);

  std::auto_ptr<unsigned int> selection ( new unsigned int() );

  // Get the number of matched b-jets in the event
  unsigned int nb = bFlavorHistoryEvent->nb();
  // Get the number of matched c-jets in the event
  unsigned int nc = cFlavorHistoryEvent->nc();
  // Get the two flavor sources. The highest takes precedence
  // over the rest. 
  FlavorHistory::FLAVOR_T bFlavorSource = bFlavorHistoryEvent->flavorSource();
  FlavorHistory::FLAVOR_T cFlavorSource = cFlavorHistoryEvent->flavorSource();
  FlavorHistory::FLAVOR_T flavorSource = FlavorHistory::FLAVOR_NULL;
  // Get the highest flavor in the event
  unsigned int highestFlavor = 0;  
  // Get the delta r between the two heavy flavor matched jets.
  double dr = -1;

  // Preference is in increasing priority:
  //  1: gluon splitting
  //  2: flavor excitation
  //  3: flavor creation (matrix element)
  //  4: flavor decay
  if ( bFlavorSource >= cFlavorSource ) {    
    flavorSource = bFlavorHistoryEvent->flavorSource();
    highestFlavor = bFlavorHistoryEvent->highestFlavor();
    dr = bFlavorHistoryEvent->deltaR();    
  }
  else {
    flavorSource = cFlavorHistoryEvent->flavorSource();
    highestFlavor = cFlavorHistoryEvent->highestFlavor();
    dr = cFlavorHistoryEvent->deltaR();
  }

  
  *selection = 0;
  // Now make hierarchical determination
  if      ( bb_me_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 1;
  else if (  b_me_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 2;
  else if ( cc_me_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 3;
  else if (  c_me_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 4;
  else if (  b_ps_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 5;
  else if (  c_ps_     ->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 6;
  else if ( bb_me_comp_->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 7;
  else if ( cc_me_comp_->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 8;
  else if (  b_ps_comp_->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 9;
  else if (  c_ps_comp_->select( nb, nc, highestFlavor, flavorSource, dr ) ) *selection = 10;
  else *selection = 11;

  bool pass = false;
  if ( pathToSelect_ > 0 ) {
    pass = (*selection > 0 && *selection == static_cast<unsigned int>(pathToSelect_ ) );
  } else {
    pass = true;
  }

  iEvent.put( selection );

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

//define this as a plug-in
DEFINE_FWK_MODULE(FlavorHistoryFilter);
