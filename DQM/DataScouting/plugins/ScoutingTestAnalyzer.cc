// This class is used to test the functionalities of the package

#include "DQM/DataScouting/plugins/ScoutingTestAnalyzer.h"


#include "DataFormats/JetReco/interface/CaloJet.h"

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
ScoutingTestAnalyzer::ScoutingTestAnalyzer( const edm::ParameterSet & conf )
   :ScoutingAnalyzerBase(conf){
  m_pfJetsCollectionTag = conf.getUntrackedParameter<edm::InputTag>("pfJetsCollectionName");
  }

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
ScoutingTestAnalyzer::~ScoutingTestAnalyzer(){}

//------------------------------------------------------------------------------
// Usual analyze method
void ScoutingTestAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){
  
  
  edm::Handle<reco::CaloJetCollection> calojets_handle ;
  iEvent.getByLabel(m_pfJetsCollectionTag,calojets_handle) ;
  /* This is an example of how C++11 can simplify or lifes. The auto keyword 
   make the compiler figure out by itself which is the type of the pfjets object.
   The qualifier const of course still apply.
   Poor's man explaination: "compiler, make pfjets a const ref and figure out 
   for me the type"*/
  auto const& calojets = *calojets_handle;
  
  // Again, C++11. A loop on a std::vector becomes as simple as this!
  for (auto const & calojet: calojets){
    m_jetPt->Fill(calojet.pt());
    m_jetEtaPhi->Fill(calojet.eta(),calojet.phi());
  }
  
  
}

//------------------------------------------------------------------------------
/* Method called at the end of the Run. Ideal to finalise stuff within the 
 * DQM infrastructure, which is entirely Run based. */
void ScoutingTestAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
    
    std::string collection_name = m_pfJetsCollectionTag.label();
    /* This function is specific of this class and allows us to make a 
     * projection in one line */
    
    profileX(m_jetEtaPhi,
             collection_name+" Jets #eta (projection)",
             "#eta^{Jet}");
    
    profileY(m_jetEtaPhi,
             collection_name+" Jets phi (projection)",
             "#phi^{Jet}");

    
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void ScoutingTestAnalyzer::bookMEs(){
  std::string collection_name = m_pfJetsCollectionTag.label();

  /* This method allows us to book an Histogram in one line in a completely 
   * transparent way. Take your time to put axis titles!!!!*/
  m_jetPt = bookH1withSumw2(collection_name+"_pt",
                            collection_name+" Jet P_{T}",
                            50,0.,500.,
                            "Jet P_{T} [GeV]");

  m_jetEtaPhi = bookH2withSumw2(collection_name+"_etaphi",
                                collection_name+" #eta #phi",
                                50,-5,5,
                                50,-3.1415,+3.1415,
                                "#eta^{Jet}",
                                "#phi^{Jet}");
}

//------------------------------------------------------------------------------

