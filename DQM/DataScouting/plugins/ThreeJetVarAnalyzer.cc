#include "DQM/DataScouting/plugins/ThreeJetVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <cmath>

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
ThreeJetVarAnalyzer::ThreeJetVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  jetPtCut_              (conf.getParameter<double>("jetPtCut")),
  htCut_                 (conf.getParameter<double>("htCut")),
  delta_                 (conf.getParameter<double>("delta")),
  jetPtCollectionTag_    (conf.getUntrackedParameter<edm::InputTag>("jetPtCollectionTag")),
  tripPtCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("tripPtCollectionTag")),
  tripMassCollectionTag_ (conf.getUntrackedParameter<edm::InputTag>("tripMassCollectionTag")){
}

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
ThreeJetVarAnalyzer::~ThreeJetVarAnalyzer(){}

//------------------------------------------------------------------------------
// Usual analyze method
void ThreeJetVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){

  edm::Handle<std::vector<double> > jetPtvar_handle;
  edm::Handle<std::vector<double> > tripMassvar_handle;
  edm::Handle<std::vector<double> > tripSumPtvar_handle;
  iEvent.getByLabel(jetPtCollectionTag_,jetPtvar_handle);
  iEvent.getByLabel(tripPtCollectionTag_,tripSumPtvar_handle);
  iEvent.getByLabel(tripMassCollectionTag_,tripMassvar_handle);

  me_Njets->Fill(jetPtvar_handle->size());
  if(jetPtvar_handle->size() > 5 && tripMassvar_handle->size() > 19 && tripSumPtvar_handle->size() > 19){
    const double lowJetpt = jetPtvar_handle->at(5);
    double Ht = 0.0;
    for (int i =0; i < int(jetPtvar_handle->size()); ++i) Ht += jetPtvar_handle->at(i);
    me_Ht->Fill(Ht);
    me_sixthJetPt->Fill(lowJetpt);
    if (lowJetpt > jetPtCut_ && Ht > htCut_){
      for (int i = 0; i < 20; ++i){
	double tripMass= tripMassvar_handle->at(i);
	double tripPt  = tripSumPtvar_handle->at(i);
	me_TripMassvsTripPt->Fill(tripPt,tripMass);
	if (tripMass < tripPt - delta_) me_TripMass->Fill(tripMass);
      }
    }
  }
}

void ThreeJetVarAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void ThreeJetVarAnalyzer::bookMEs(){

  //6th jet pT
  me_sixthJetPt = bookH1withSumw2("sixthJetPt",
				  "6^{TH} jet p_{T} (GeV)",
				  250,0.,500.,
				  "6^{TH} jet p_{T} (GeV)");
  //HT distribution
  me_Ht         = bookH1withSumw2("Ht",
				  "H_{T}",
				  100, 0.,2000.,
				  "H_{T} (GeV)");
  //Njet distribution
  me_Njets      = bookH1withSumw2("Njets",
				  "Number of Jets",
				  16,0.,16.,
				  "N_{JETS} / Event");
  //triplet mass distribution
  me_TripMass   = bookH1withSumw2("TripMass",
				  "Three-Jet Mass",
				  1500,0.,3000.,
				  "Triplet Mass M_{jjj} (GeV)");
  //2D triplet pt vs triplet mass
  me_TripMassvsTripPt= bookH2withSumw2("TripMassvsTripPt",
				       "Triplet Mass M_{jjj} vs Triplet scalar p_{T}",
				       150,0.,3000.,
				       150,0.,3000.,
				       "Triplet Mass M_{jjj} (GeV)",
				       "Triplet scalar p_{T} (GeV)");
}

