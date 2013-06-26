#include "DQM/DataScouting/plugins/AlphaTVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include <cmath>

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
AlphaTVarAnalyzer::AlphaTVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  m_jetCollectionTag(conf.getUntrackedParameter<edm::InputTag>("jetCollectionName",edm::InputTag("hltCaloJetIDPassed"))),
  m_alphaTVarCollectionTag(conf.getUntrackedParameter<edm::InputTag>("alphaTVarCollectionName")){
}

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
AlphaTVarAnalyzer::~AlphaTVarAnalyzer(){}

//------------------------------------------------------------------------------
// Usual analyze method
void AlphaTVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){
  
  edm::Handle<std::vector<double> > alphaTvar_handle;
  iEvent.getByLabel(m_alphaTVarCollectionTag,alphaTvar_handle);

  if(alphaTvar_handle->size() > 1){
    const double AlphaT = alphaTvar_handle->at(0);
    const double HT = alphaTvar_handle->at(1);
    m_HTAlphaT->Fill(HT,AlphaT);
    if(AlphaT > 0.55) m_HTAlphaTg0p55->Fill(HT);
    if(AlphaT > 0.60) m_HTAlphaTg0p60->Fill(HT);
    
  }
  
}

void AlphaTVarAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void AlphaTVarAnalyzer::bookMEs(){
  
  //the full inclusive histograms
  m_HTAlphaT = bookH2withSumw2("HTvsAlphaT",
			       "H_{T} vs #alpha_{T} (All Events)",
			       400,0.,4000.,
			       50,0.,1.,
			       "H_{T} [GeV]",
			       "#alpha_{T}");
  m_HTAlphaTg0p55 = bookH1withSumw2("HTvsAlphaTg0p55",
				   "H_{T} (#alpha_{T} > 0.55)",
				   400,0.,4000.,
				   "H_{T} [GeV]");

  m_HTAlphaTg0p60 = bookH1withSumw2("HTvsAlphaTg0p60",
				   "H_{T} (#alpha_{T} > 0.60)",
				   400,0.,4000.,
				   "H_{T} [GeV]");
}

