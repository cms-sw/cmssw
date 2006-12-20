// CorJetsExample.cc
// Description:  Example of simple EDAnalyzer for jets.
// Author: Robert M. Harris
// Date:  28 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/CorJetsExample.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;

// Get the algorithm of the jet collections we will read from the .cfg file 
// which defines the value of the strings CaloJetAlgorithm and GenJetAlgorithm.
CorJetsExample::CorJetsExample( const ParameterSet & cfg ) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) ), 
  CorJetAlgorithm( cfg.getParameter<string>( "CorJetAlgorithm" ) ), 
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) )
  {
}

void CorJetsExample::beginJob( const EventSetup & ) {

  // Open the histogram file and book some associated histograms
  m_file=new TFile("histo.root","RECREATE"); 
  h_ptCal =  TH1F( "ptCal",  "p_{T} of leading CaloJets", 50, 0, 1000 );
  h_ptGen =  TH1F( "ptGen",  "p_{T} of leading GenJets", 50, 0, 1000 );
  h_ptCor =  TH1F( "ptCor",  "p_{T} of leading CorJets", 50, 0, 1000 );
}

void CorJetsExample::analyze( const Event& evt, const EventSetup& es ) {

  //Get the CaloJet collection
  Handle<CaloJetCollection> caloJets;
  evt.getByLabel( CaloJetAlgorithm, caloJets );

  //Loop over the two leading CaloJets and fill a histogram
  int jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    h_ptCal.Fill( cal->pt() );   
    jetInd++;
  }

  //Get the GenJet collection
  Handle<GenJetCollection> genJets;
  evt.getByLabel( GenJetAlgorithm, genJets );

  //Loop over the two leading GenJets and fill a histogram
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    h_ptGen.Fill( gen->pt() );   
    jetInd++;
  }

  //Get the Corrected Jet collection
  Handle<CaloJetCollection> corJets;
  evt.getByLabel( CorJetAlgorithm, corJets );

  //Loop over the two leading CorJets and fill a histogram
  jetInd = 0;
  for( CaloJetCollection::const_iterator cor = corJets->begin(); cor != corJets->end() && jetInd<2; ++ cor ) {
    h_ptCor.Fill( cor->pt() );   
    jetInd++;
  }


}

void CorJetsExample::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
