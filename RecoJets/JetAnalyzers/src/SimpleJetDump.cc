// SimpleJetDump.cc
// Description:  Prints out Jets, consituent CaloTowers, constituent RecHits and associated Digis (the digis for HCAL only).
//               The user can specify which level in the config file:
//               DumpLevel="Jets":    Printout of jets and their kinematic quantities.
//               DumpLevel="Towers":  Nested Printout of jets and their constituent CaloTowers
//               DumpLevel="RecHits": Nested Printout of jets, constituent CaloTowers and constituent RecHits
//               DumpLevel="Digis":   Nested Printout of jets, constituent CaloTowers, RecHits and all the HCAL digis 
//                                    associated with the RecHit channel (no links exist to go back to actual digis used).
//               Does simple sanity checks on energy sums at each level: jets=sum of towers, tower=sum of RecHits.
//               Does quick and dirty estimate of the fC/GeV factor that was applied to make the RecHit from the Digis.
//               
// Author: Robert M. Harris
// Date:  19 - October - 2006
// 
#include "RecoJets/JetAnalyzers/interface/SimpleJetDump.h"
#include "RecoJets/JetAnalyzers/interface/JetPlotsExample.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace reco;
using namespace std;

SimpleJetDump::SimpleJetDump( const ParameterSet & cfg ) :  
  CaloJetAlg( cfg.getParameter<string>( "CaloJetAlg" ) ),
  GenJetAlg( cfg.getParameter<string>( "GenJetAlg" ) )  
  {
}

void SimpleJetDump::beginJob( ) {
  evtCount = 0;
}

void SimpleJetDump::analyze( const Event& evt, const EventSetup& es ) {

  int jetInd;
  Handle<CaloJetCollection> caloJets;
  Handle<GenJetCollection> genJets;
   
  //Find the CaloTowers in leading CaloJets
  evt.getByLabel( CaloJetAlg, caloJets );
  evt.getByLabel( GenJetAlg, genJets );
    
  cout << endl << "Evt: "<<evtCount <<", Num Calo Jets=" <<caloJets->end() - caloJets->begin() << ", Num Gen Jets=" <<genJets->end() - genJets->begin() <<endl;
  cout <<"   *********************************************************" <<endl;
  jetInd = 0;
  for( CaloJetCollection::const_iterator jet = caloJets->begin(); jet != caloJets->end(); ++ jet ) {
    cout <<"Calo Jet: "<<jetInd<<", pt="<<jet->pt()<<", eta="<<jet->eta()<<", phi="<<jet->phi() <<endl;
    jetInd++;
  }
  cout <<"   *********************************************************" <<endl;
  jetInd = 0;
  for( GenJetCollection::const_iterator jet = genJets->begin(); jet != genJets->end(); ++ jet ) {
    cout <<"Gen Jet: "<<jetInd<<", pt="<<jet->pt()<<", eta="<<jet->eta()<<", phi="<<jet->phi() <<endl;
    jetInd++;
  }
  evtCount++;    
  cout <<"   *********************************************************" <<endl;

}

void SimpleJetDump::endJob() {


}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleJetDump);
