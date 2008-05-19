#include "PhysicsTools/StarterKit/interface/ZGammaJetBalance.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
//
// constructors and destructor
//
ZGammaJetBalanceKit::ZGammaJetBalanceKit(const edm::ParameterSet& iConfig)
  :
  StarterKit        ( iConfig ),
  zHandleName_ ( iConfig.getParameter<edm::InputTag> ("zHandleName") )
{

  verboseLevel_ = 10;


  // NOTE: These are hard-coded for now, change to something meaningful in future

  // Composite histograms
  zHistos_ = new pat::HistoComposite(zHandleName_.label(), 
				     "Z Histograms",
				     zHandleName_.label(),
				     0, 200, 0, 200 );
  // Slight kludge until we get expression histograms working

  // Make service directory
  edm::Service<TFileService> fs;
  TFileDirectory res = TFileDirectory( fs->mkdir("zgammajet") );

  ptJet1OverPtZ_ = new pat::PhysVarHisto( "ptJet1OverPtZ", "Pt of Jet 1 over Pt of Z",
					  100, 0, 10, &res, "", "vD" ); 
  ptJet1OverPtGamma_ = new pat::PhysVarHisto( "ptJet1OverPtGamma", "Pt of Jet 1 over Pt of Gamma",
					      100, 0, 10, &res, "", "vD" );
  // make associated TH1
  ptJet1OverPtZ_->makeTH1();
  ptJet1OverPtGamma_->makeTH1();


  // ----- Name production branch -----
  string list_of_ntuple_vars =
    iConfig.getParameter<std::string>    ("ntuplize");
  
  if ( list_of_ntuple_vars != "" ) {

    // add resonance mass to list of variables to ntuplize
    zgammaNtVars_.push_back( ptJet1OverPtZ_ );
    zgammaNtVars_.push_back( ptJet1OverPtGamma_ );

    //--- Iterate over the list and "book" them via EDM
    std::vector< pat::PhysVarHisto* >::iterator
      p    = zgammaNtVars_.begin(),
      pEnd = zgammaNtVars_.end();

    for ( ; p != pEnd; ++p ) {
      addNtupleVar( (*p)->name(), (*p)->type() );
    }
  }
}


ZGammaJetBalanceKit::~ZGammaJetBalanceKit()
{
  if ( ptJet1OverPtZ_ ) delete ptJet1OverPtZ_;
  if ( ptJet1OverPtGamma_ ) delete ptJet1OverPtGamma_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void ZGammaJetBalanceKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  if ( jetHandle_->size() > 0 ) {

    // Get the first jet, sorted by pt
    pat::Jet const & ijet = jetHandle_->at(0);

    bool found = false;

    if ( muonHandle_->size() >= 2 || electronHandle_->size() >= 2 ) {
      // Get the composite candidates from upstream
      iEvent.getByLabel(zHandleName_,   zHandle_ );

      // Try Z + jet case first
      if ( zHandle_->size() > 0 ) {
	// Loop over the input Z collection
	vector<NamedCompositeCandidate>::const_iterator i = zHandle_->begin(),
	  iend = zHandle_->end();
	for ( ; i != iend; ++i ) {
	  // Fill Z histograms
	  zHistos_->fill( *i );
	  // Get the Pt of the Z
	  double ptZ = i->pt();
	  // Plot the Pt of the jet over Pt of the Z
	  if ( ptZ > 0 ) {
	    found = true;
	    double f = ijet.pt() / ptZ;
	    ptJet1OverPtZ_->fill( f );
	  }
	}
    
      }
    } 
    // Try gamma + jet case second
    else if ( photonHandle_->size() > 0 ) {
      vector<pat::Photon>::const_iterator i = photonHandle_->begin(),
	iend = photonHandle_->end();
      for( ; i != iend; ++i ) {
	double ptG = i->pt();
	if ( ptG > 0 ) {
	  found = true;
	  double f = ijet.pt() / ptG;
	  ptJet1OverPtGamma_->fill( f );
	}
      }
    }
    // else do nothing

    
    if ( found ) {
      // Save the ntuple variables
      saveNtuple( zgammaNtVars_, iEvent );

      // Clear the ntuple cache
      ptJet1OverPtZ_->clearVec();
      ptJet1OverPtGamma_->clearVec();
    }
  }
  
}


// ------------ method called once each job just before starting event loop  ------------
void
ZGammaJetBalanceKit::beginJob(const edm::EventSetup& iSetup)
{
  StarterKit::beginJob(iSetup);
}



// ------------ method called once each job just after ending the event loop  ------------
void
ZGammaJetBalanceKit::endJob() {
  StarterKit::endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZGammaJetBalanceKit);
