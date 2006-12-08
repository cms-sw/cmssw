// JetPlotsExample.cc
// Description:  Example of simple EDAnalyzer for jets.
// Author: Robert M. Harris
// Date:  28 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/JetPlotsExample.h"
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
JetPlotsExample::JetPlotsExample( const ParameterSet & cfg ) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) ), 
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) )
  {
}

void JetPlotsExample::beginJob( const EventSetup & ) {

  // Open the histogram file and book some associated histograms
  m_file=new TFile("histo.root","RECREATE"); 
  h_ptCal =  TH1F( "ptCal",  "p_{T} of leading CaloJets", 50, 0, 1000 );
  h_etaCal = TH1F( "etaCal", "#eta of leading CaloJets", 50, -3, 3 );
  h_phiCal = TH1F( "phiCal", "#phi of leading CaloJets", 50, -M_PI, M_PI );
  h_ptGen =  TH1F( "ptGen",  "p_{T} of leading GenJets", 50, 0, 1000 );
  h_etaGen = TH1F( "etaGen", "#eta of leading GenJets", 50, -3, 3 );
  h_phiGen = TH1F( "phiGen", "#phi of leading GenJets", 50, -M_PI, M_PI );
}

void JetPlotsExample::analyze( const Event& evt, const EventSetup& es ) {

  //Get the CaloJet collection
  Handle<CaloJetCollection> caloJets;
  evt.getByLabel( CaloJetAlgorithm, caloJets );

  //Loop over the two leading CaloJets and fill some histograms
  int jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    // std::cout << "CALO JET #" << jetInd << std::endl << cal->print() << std::endl;
    h_ptCal.Fill( cal->pt() );   
    h_etaCal.Fill( cal->eta() );
    h_phiCal.Fill( cal->phi() );
    jetInd++;
  }

  //Get the GenJet collection
  Handle<GenJetCollection> genJets;
  evt.getByLabel( GenJetAlgorithm, genJets );

  //Loop over the two leading GenJets and fill some histograms
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    // std::cout << "GEN JET #" << jetInd << std::endl << gen->print() << std::endl;
    h_ptGen.Fill( gen->pt() );   
    h_etaGen.Fill( gen->eta() );
    h_phiGen.Fill( gen->phi() );
    jetInd++;
  }

}

void JetPlotsExample::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
