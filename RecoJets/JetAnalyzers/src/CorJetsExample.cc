// CorJetsExample.cc
// Description:  Example of simple EDAnalyzer for jets.
// Author: Robert M. Harris
// Date:  28 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/CorJetsExample.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
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
  JetCorrectionService( cfg.getParameter<string>( "JetCorrectionService" ) ), 
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) )
  {
}

void CorJetsExample::beginJob( const EventSetup & ) {

  // Open the histogram file and book some associated histograms
  m_file=new TFile("histo.root","RECREATE"); 
  h_ptCal =  TH1F( "h_ptCal",  "p_{T} of leading CaloJets", 50, 0, 1000 );
  h_ptGen =  TH1F( "h_ptGen",  "p_{T} of leading GenJets", 50, 0, 1000 );
  h_ptCor =  TH1F( "h_ptCor",  "p_{T} of leading CorJets", 50, 0, 1000 );
  h_ptCorOnFly =  TH1F( "h_ptCorOnFly",  "p_{T} of leading Jets Corrected on the Fly", 50, 0, 1000 );
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

  //Loop over all the CorJets, save the two highest Pt, and fill the histogram.
  //Corrected jets have the original Pt order as the uncorrected CaloJets for releases before 1.5.0,
  // and hence need to be re-ordered to find the two leading jets.
  jetInd = 0;
  double highestPt=0.0;
  double nextPt=0.0;
  for( CaloJetCollection::const_iterator cor = corJets->begin(); cor != corJets->end(); ++ cor ) {
    double corPt=cor->pt();
    //std::cout << "cor=" << cor->pt() << std::endl;
    if(corPt>highestPt){
      nextPt=highestPt;
      highestPt=corPt;
    }
    else if(corPt>nextPt)nextPt=corPt;
  }
  h_ptCor.Fill( highestPt );   
  h_ptCor.Fill( nextPt );

  const JetCorrector* corrector = 
                 JetCorrector::getJetCorrector (JetCorrectionService, es);
  //Loop over the CaloJets, correct them on the fly, save the two highest Pt and fill the histogram.
  highestPt=0.0;
  nextPt=0.0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
    double scale = corrector->correction (*cal);
    double corPt=scale*cal->pt();
    //std::cout << corPt << ", scale=" << scale << std::endl;  
    if(corPt>highestPt){
      nextPt=highestPt;
      highestPt=corPt;
    }
    else if(corPt>nextPt)nextPt=corPt;
  }
  h_ptCorOnFly.Fill( highestPt );   
  h_ptCorOnFly.Fill( nextPt );
  //std::cout <<  "corOnFly=" << highestPt  << std::endl;
  //std::cout <<  "corOnFly=" << nextPt  << std::endl;


}

void CorJetsExample::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CorJetsExample);
