#include "PhysicsTools/StarterKit/interface/CompositeKit.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace reco;
//
// constructors and destructor
//
CompositeKit::CompositeKit(const edm::ParameterSet& iConfig)
  :
  StarterKit        ( iConfig ),
  compositeCandTag_ ( iConfig.getParameter<edm::InputTag> ("compositeCandTag") ),
  description_      ( iConfig.getParameter<std::string>   ("description") ),
  pt1_              ( iConfig.getParameter<double>        ("pt1") ),
  pt2_              ( iConfig.getParameter<double>        ("pt2") ),
  m1_               ( iConfig.getParameter<double>        ("m1") ),
  m2_               ( iConfig.getParameter<double>        ("m2") ),
  mm1_              ( iConfig.getParameter<double>        ("resonanceM1") ),
  mm2_              ( iConfig.getParameter<double>        ("resonanceM2") )
{

  verboseLevel_ = 10;


  // NOTE: These are hard-coded for now, change to something meaningful in future
  std::string alias;
  compositeCandMassName_ = std::string( compositeCandTag_.label() );
  compositeCandMassName_.append( "ResonanceMass");

  // Composite histograms
  compositeCandHist_ = new pat::HistoComposite(compositeCandTag_.label(), 
					       description_,
					       compositeCandTag_.label(),
					       pt1_,pt2_,m1_,m2_ );
  // Slight kludge until we get expression histograms working

  // Make service directory
  edm::Service<TFileService> fs;
  TFileDirectory res = TFileDirectory( fs->mkdir("resonance") );
  // Make resonance mass histogram
  compositeCandMass_ = new pat::PhysVarHisto( compositeCandMassName_, description_, 
					      20, mm1_, mm2_, &res, "", "vD" );
  // make associated TH1
  compositeCandMass_->makeTH1();


  // ----- Name production branch -----
  string list_of_ntuple_vars =
    iConfig.getParameter<std::string>    ("ntuplize");
  
  if ( list_of_ntuple_vars != "" ) {

    // add resonance mass to list of variables to ntuplize
    compositeNtVars_.push_back( compositeCandMass_ );

    //--- Iterate over the list and "book" them via EDM
    std::vector< pat::PhysVarHisto* >::iterator
      p    = compositeNtVars_.begin(),
      pEnd = compositeNtVars_.end();

    for ( ; p != pEnd; ++p ) {
      addNtupleVar( (*p)->name(), (*p)->type() );
    }
  }
}


CompositeKit::~CompositeKit()
{
  if ( compositeCandMass_ ) delete compositeCandMass_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void CompositeKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  StarterKit::produce( iEvent, iSetup );


  // INSIDE OF StarterKit::analyze:

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------



  // BEGIN YOUR CODE HERE

  // --------------------------------------------------
  //    Step 3: Plot some composite objects
  // --------------------------------------------------

  // Get the composite candidates from upstream
  iEvent.getByLabel(compositeCandTag_,   compositeCandHandle_ );

  if ( compositeCandHandle_->size() > 0 ) {

    // Get the vector of masses... when expression histograms come along, this 
    // will disappear
    vector<double> compositeCandMassVector;

    // Loop over the composite candidates
    vector<NamedCompositeCandidate>::const_iterator i = compositeCandHandle_->begin(),
      iend = compositeCandHandle_->end();
    for ( ; i != iend; ++i ) {
      compositeCandHist_->fill( *i );
      compositeCandMass_->fill( i->mass() );
    }
    
    // Save the ntuple variables... in this case just the mass
    saveNtuple( compositeNtVars_, iEvent );

    // Clear the ntuple cache... in this case just the mass
    compositeCandMass_->clearVec();
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
CompositeKit::beginJob(const edm::EventSetup& iSetup)
{
  StarterKit::beginJob(iSetup);
}



// ------------ method called once each job just after ending the event loop  ------------
void
CompositeKit::endJob() {
  StarterKit::endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(CompositeKit);
