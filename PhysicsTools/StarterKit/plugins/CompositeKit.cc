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
  src_              ( iConfig.getParameter<edm::InputTag> ("src") ),
  description_      ( iConfig.getParameter<std::string>   ("description") ),
  helper_           ( iConfig )
{

  verboseLevel_ = 0;


  // First book standard histograms
  helper_.bookHistos(this);

  PhysicsHistograms::KinAxisLimits compositeAxisLimits;

  compositeAxisLimits = helper_.getAxisLimits("compositeAxis");

  double pt1 = compositeAxisLimits.pt1;
  double pt2 = compositeAxisLimits.pt2;
  double m1  = compositeAxisLimits.m1;
  double m2  = compositeAxisLimits.m2;

  // Now book composite histograms
  compositeCandHist_ = new pat::HistoComposite(src_.label(), 
					       description_,
					       src_.label(),
					       pt1,pt2,m1,m2 );


}


CompositeKit::~CompositeKit()
{
  // clean up
  if ( compositeCandHist_ ) delete compositeCandHist_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void CompositeKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------
  helper_.getHandles( iEvent,
		      muonHandle_,
		      electronHandle_,
		      tauHandle_,
		      jetHandle_,
		      METHandle_,
		      photonHandle_);

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------

  if ( verboseLevel_ > 10 )
    std::cout << "PatAnalyzerKit::analyze: calling fillCollection()." << std::endl;
  helper_.fillHistograms( iEvent,
			  muonHandle_,
			  electronHandle_,
			  tauHandle_,
			  jetHandle_,
			  METHandle_,
			  photonHandle_);


  // BEGIN YOUR CODE HERE

  // --------------------------------------------------
  //    Step 3: Plot some composite objects
  // --------------------------------------------------

  // Get the composite candidates from upstream
  iEvent.getByLabel(src_,   compositeCandHandle_ );

  if ( compositeCandHandle_->size() > 0 ) {

    // Get the vector of masses... when expression histograms come along, this 
    // will disappear
    vector<double> compositeCandMassVector;

    // Loop over the composite candidates
    vector<reco::CompositeCandidate>::const_iterator i = compositeCandHandle_->begin(),
      iend = compositeCandHandle_->end();
    for ( ; i != iend; ++i ) {
      compositeCandHist_->fill( *i );
    }
    
    // Save the ntuple variables... in this case just the mass
//     saveNtuple( compositeNtVars_, iEvent );

  }
}


// ------------ method called once each job just before starting event loop  ------------
void
CompositeKit::beginJob(const edm::EventSetup& iSetup)
{
}



// ------------ method called once each job just after ending the event loop  ------------
void
CompositeKit::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(CompositeKit);
