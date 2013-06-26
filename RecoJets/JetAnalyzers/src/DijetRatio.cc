// -*- C++ -*-
//
// Package:    DijetRatio
// Class:      DijetRatio
// 
/**\class DijetRatio DijetRatio.cc RecoJets/DijetRatio/src/DijetRatio.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Manoj Jha
//         Created:  Thu Apr 12 15:04:37 CDT 2007
// Kalanand Mishra (November 22, 2009): 
//    Modified and cleaned up to work in 3.3.X
//
//

#include "RecoJets/JetAnalyzers/interface/DijetRatio.h"

template<class Jet>
DijetRatio<Jet>::DijetRatio(const edm::ParameterSet& iConfig)
{
   
   //get name of output file with histograms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistoFileName", 
							  "DijetRatio.root"); 
  
  // get names of modules, producing object collections
  m_Mid5CorRecJetsSrc   = iConfig.getParameter<std::string>("CorrectedJets");
  m_Mid5CaloJetsSrc   = iConfig.getParameter<std::string>("UnCorrectedJets");
  
  // eta limit for numerator and denominator
  m_eta3   = iConfig.getUntrackedParameter<double>("etaInner", 0.7);
  m_eta4   = iConfig.getUntrackedParameter<double>("etaOuter", 1.3);

}



template<class Jet>
DijetRatio<Jet>::~DijetRatio()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------

template<class Jet>
void DijetRatio<Jet>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   // get calo jet collection
   Handle<JetCollection> Mid5CorRecJets;
   iEvent.getByLabel(m_Mid5CorRecJetsSrc, Mid5CorRecJets);
   
   Handle<JetCollection> Mid5CaloJets;
   iEvent.getByLabel(m_Mid5CaloJetsSrc, Mid5CaloJets);

   histoFill(hCalo, Mid5CaloJets, m_eta3, m_eta4);
   histoFill(hCor, Mid5CorRecJets, m_eta3, m_eta4);
}



// ------------ method called once each job just before starting event loop  ------------
template <class Jet>
void  DijetRatio<Jet>::beginJob()
{
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

   // Histo Initializations for  Jets
   hInit(hCalo, "DijetRatio_UnCorrectedJets");
   hInit(hCor, "DijetRatio_CorrectedJets");

   return;
}

// ------------ method called once each job just after ending the event loop  ------------

template<class Jet>
void  DijetRatio<Jet>::endJob() {
  
  hOutputFile->cd();
  for(int i=0; i<histoSize; ++i) {
    hCalo[i]->Write() ;
    hCor[i]->Write() ;
  }
  hOutputFile->Close() ;

return ;
}

#include "FWCore/Framework/interface/MakerMacros.h"
/////////// Calo Jet Instance ////////
typedef DijetRatio<reco::CaloJet> DijetRatioCaloJets;
DEFINE_FWK_MODULE(DijetRatioCaloJets);
/////////// Gen Jet Instance ////////
typedef DijetRatio<reco::GenJet> DijetRatioGenJets;
DEFINE_FWK_MODULE(DijetRatioGenJets);
/////////// PF Jet Instance ////////
typedef DijetRatio<reco::PFJet> DijetRatioPFJets;
DEFINE_FWK_MODULE(DijetRatioPFJets);
