// CaloTowersExample.cc
// Description:  Example of simple EDAnalyzer for CaloTowers.
// Author: Robert M. Harris
// Date:  8 - September - 2006
// 
#include "RecoJets/JetAnalyzers/interface/CaloTowersExample.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>
using namespace edm;
using namespace std;

// Get the algorithm of the jet collections we will read from the .cfg file 
// which defines the value of the strings CaloJetAlgorithm and GenJetAlgorithm.
CaloTowersExample::CaloTowersExample( const ParameterSet & cfg ) :
  CaloTowersAlgorithm( cfg.getParameter<string>( "CaloTowersAlgorithm" ) )
  {
}

void CaloTowersExample::beginJob( ) {

  // Open the histogram file and book some associated histograms
  m_file=new TFile("histo.root","RECREATE"); 
  h_et =  TH1F( "et",  "E_{T} of leading CaloTowers", 50, 0, 25 );
}

void CaloTowersExample::analyze( const Event& evt, const EventSetup& es ) {

  //Get the CaloTower collection
  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( CaloTowersAlgorithm, caloTowers );

  //Loop over the two leading CaloJets and fill some histograms
  for( CaloTowerCollection::const_iterator cal = caloTowers->begin(); cal != caloTowers->end(); ++ cal ) {
    h_et.Fill( cal->et() );   
  }

}

void CaloTowersExample::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloTowersExample);
