#include "PhysicsTools/StarterKit/interface/HZZKit.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
//
// constructors and destructor
//
HZZKit::HZZKit(const edm::ParameterSet& iConfig)
  :
  StarterKit( iConfig )
{

  verboseLevel_ = 10;

  // Make some personal histograms by hand

  // Composite if there are only 2 muons
  zmumuCandHist_ = new pat::HistoComposite("zmumuCand", "Z to mu mu Candidates",    "zmumu",
					   0., 200., 0., 200. );
  // Composite if there are only two electrons
  zeeCandHist_   = new pat::HistoComposite("zeeCand",   "Z to mu mu Candidates",    "zee",
					   0., 200., 0., 200. );
  // Composite if there are any 4 leptons
  hCandHist_     = new pat::HistoComposite("hCand",     "Higgs to Z Z Candidates",  "h",
					   0., 200., 0., 300. );

  // NOTE: These are hard-coded for now, change to something meaningful in future
  std::string alias;
  produces<vector<double> >      ( alias = "zmumuMass" ).setBranchAlias( alias );
  produces<vector<double> >      ( alias = "hMass"     ).setBranchAlias( alias );
}


HZZKit::~HZZKit()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void HZZKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup)
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
  iEvent.getByLabel("zToMuMu",   zToMuMuHandle_ );
  iEvent.getByLabel("hToZZ",     hToZZHandle_ );

  vector<double> zmumuMass;
  vector<double> hMass;


  // Histogram the Z->mumu candidates
  // NOTE: This will run over SM Z production events as well
  if ( zToMuMuHandle_->size() > 0 ) {
    vector<reco::NamedCompositeCandidate>::const_iterator iz = zToMuMuHandle_->begin();
    vector<reco::NamedCompositeCandidate>::const_iterator zend = zToMuMuHandle_->end();
    for ( ; iz != zend; ++iz ) {
      zmumuCandHist_->fill( &*iz );
      zmumuMass.push_back( iz->mass() );
    }
  }

  // Histogram the H->ZZ candidates
  vector<reco::NamedCompositeCandidate>::const_iterator ih = hToZZHandle_->begin();
  for ( ; ih != hToZZHandle_->end(); ih++ ) {
    hCandHist_->fill( &*ih );
    hMass.push_back( ih->mass() );
  }

  // Ntuplize the masses
  std::auto_ptr<vector<double> > ap_zmass( new vector<double> ( zmumuMass ) );
  std::auto_ptr<vector<double> > ap_hmass( new vector<double> ( hMass ) );  
  iEvent.put( ap_zmass, "zmumuMass" );
  iEvent.put( ap_hmass, "hMass" );

  zmumuMass.clear();
  hMass.clear();

}


// ------------ method called once each job just before starting event loop  ------------
void
HZZKit::beginJob(const edm::EventSetup& iSetup)
{
  StarterKit::beginJob(iSetup);
}



// ------------ method called once each job just after ending the event loop  ------------
void
HZZKit::endJob() {
  StarterKit::endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HZZKit);
