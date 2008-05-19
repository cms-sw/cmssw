#include "PhysicsTools/StarterKit/interface/TtSemiEvtKit.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace pat;

//
// constructors and destructor
//
TtSemiEvtKit::TtSemiEvtKit(const edm::ParameterSet& iConfig) 
  :
  LepJetMetKit( iConfig )
{


  evtsols           = iConfig.getParameter<edm::InputTag> ("EvtSolution");

  cout << "About to book histoTtSemiEvtHypothesis" << endl;
  histoTtSemiEvt_ = new HistoComposite("ttSemiEvt", "ttSemiEvt", "ttSemiEvt");


  edm::Service<TFileService> fs;
  TFileDirectory ttbar = TFileDirectory( fs->mkdir("ttbar") );

  histoHadb_ = new HistoJet("ttbar", "hadb", "hadb");
  histoHadq_ = new HistoJet("ttbar", "hadq", "hadq");
  histoHadp_ = new HistoJet("ttbar", "hadp", "hadp");
  histoLepb_ = new HistoJet("ttbar", "lepb", "lepb");

  histoLRJetCombProb_ = new PhysVarHisto( "lrJetCombProb", "Jet Comb Probability",
					  100, 0, 1, &ttbar, "", "vD" );
  histoLRSignalEvtProb_ = new PhysVarHisto( "lrSignalEvtProb", "Event Probability",
					  100, 0, 1, &ttbar, "", "vD" );
  histoKinFitProbChi2_ = new PhysVarHisto( "kinFitProbChi2", "Kin Fitter Chi2 Prob",
					  100, 0, 1, &ttbar, "", "vD" );
  histoTtMass_ = new PhysVarHisto( "ttMass", "ttbar invariant mass",
				   100, 0, 5000, &ttbar, "", "vD" );


  histoLRJetCombProb_ ->makeTH1();
  histoLRSignalEvtProb_ ->makeTH1();
  histoKinFitProbChi2_ ->makeTH1();
  histoTtMass_ ->makeTH1();
  

  // ----- Name production branch -----
  string list_of_ntuple_vars =
    iConfig.getParameter<std::string>    ("ntuplize");
  
  if ( list_of_ntuple_vars != "" ) {

    ttNtVars_.push_back( histoLRJetCombProb_ );
    ttNtVars_.push_back( histoLRSignalEvtProb_ );
    ttNtVars_.push_back( histoKinFitProbChi2_ );
    ttNtVars_.push_back( histoTtMass_ );

    histoHadb_->select( list_of_ntuple_vars, ttNtVars_ );
    histoHadq_->select( list_of_ntuple_vars, ttNtVars_ );
    histoHadp_->select( list_of_ntuple_vars, ttNtVars_ );
    histoLepb_->select( list_of_ntuple_vars, ttNtVars_ );

    
    //--- Iterate over the list and "book" them via EDM
    std::vector< PhysVarHisto* >::iterator
      p    = ttNtVars_.begin(),
      pEnd = ttNtVars_.end();

    for ( ; p != pEnd; ++p ) {
      addNtupleVar( (*p)->name(), (*p)->type() );
    }
  }
}

TtSemiEvtKit::~TtSemiEvtKit() 
{
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TtSemiEvtKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  using namespace edm;

//   cout << "About to produce LepJetMetKit" << endl;
  LepJetMetKit::produce( iEvent, iSetup );

  // INSIDE OF LepJetMetKit::produce:

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

  // --------------------------------------------------
  //    Step 3: Plot LepJetMet data
  // --------------------------------------------------


  // BEGIN TtSemiEvt analysis here:

   // get the event solution
   edm::Handle< std::vector<TtSemiEvtSolution> > eSols; 
   iEvent.getByLabel(evtsols, eSols);

//    cout << "TtSemiEvtKit: About to do work on sols" << endl;
   const std::vector<TtSemiEvtSolution> & sols = *eSols;
//    cout << "Done getting vector ref to sols" << endl;
  
   if ( sols.size() > 0 ) {

//      cout << "Sols.size() > 0 " << endl;

//      cout << "TtSemiEvtKit: Getting best solution" << endl;
     int bestSol = sols[0].getLRBestJetComb();   
//      cout << "About to fill the ttSemiEvt solution : " << bestSol << endl;
     histoTtSemiEvt_->fill( sols[bestSol].getRecoHyp() );
     histoHadb_->fill( sols[bestSol].getHadb() );
     histoHadq_->fill( sols[bestSol].getHadq() );
     histoHadp_->fill( sols[bestSol].getHadp() );
     histoLepb_->fill( sols[bestSol].getLepb() );

     
     histoLRJetCombProb_->fill( sols[bestSol].getLRJetCombProb());
     histoLRSignalEvtProb_->fill( sols[bestSol].getLRSignalEvtProb());
     histoKinFitProbChi2_->fill( sols[bestSol].getProbChi2());
     histoTtMass_->fill( sols[bestSol].getRecoHyp().mass() );
   }
  
   saveNtuple( ttNtVars_, iEvent );

   
   histoTtSemiEvt_->clearVec();  
   histoHadb_->clearVec();
   histoHadq_->clearVec();
   histoHadp_->clearVec();
   histoLepb_->clearVec();
   
   histoLRJetCombProb_->clearVec();
   histoLRSignalEvtProb_->clearVec();
   histoKinFitProbChi2_->clearVec();
   histoTtMass_->clearVec();


   // cout << "Done with produce" << endl;
}


// ------------ method called once each job just before starting event loop  ------------
void
TtSemiEvtKit::beginJob(const edm::EventSetup& iSetup)
{
  LepJetMetKit::beginJob(iSetup);
}



// ------------ method called once each job just after ending the event loop  ------------
void
TtSemiEvtKit::endJob() {
  LepJetMetKit::endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TtSemiEvtKit);
