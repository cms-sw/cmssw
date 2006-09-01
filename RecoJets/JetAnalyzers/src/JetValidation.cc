// JetValidation.cc
// Description:  Some Basic validation plots for jets.
// Author: Robert M. Harris
// Date:  30 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/JetValidation.h"
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
JetValidation::JetValidation( const ParameterSet & cfg ) :
  PtHistMax( cfg.getParameter<double>( "PtHistMax" ) )
  
  {
}

void JetValidation::beginJob( const EventSetup & ) {
  cout << "JetValidation: Maximum bin edge for Pt Hists = " << PtHistMax << endl;

  // Open the histogram file and book some associated histograms
  m_file=new TFile("JetValidationHistos.root","RECREATE"); 
  
  //Simple histos
  
  //MC5 cal
  ptMC5cal  = TH1F("ptMC5cal","p_{T} of leading CaloJets (MC5)",50,0.0,PtHistMax);
  etaMC5cal = TH1F("etaMC5cal","#eta of leading CaloJets (MC5)",100,-5.0,5.0);
  phiMC5cal = TH1F("phiMC5cal","#phi of leading CaloJets (MC5)",72,-M_PI, M_PI);
  m2jMC5cal = TH1F("m2jMC5cal","Dijet Mass of leading CaloJets (MC5)",100,0.0,2*PtHistMax);

  //MC5 gen
  ptMC5gen = TH1F("ptMC5gen","p_{T} of leading GenJets (MC5)",50,0.0,PtHistMax);
  etaMC5gen = TH1F("etaMC5gen","#eta of leading GenJets (MC5)",100,-5.0,5.0);
  phiMC5gen = TH1F("phiMC5gen","#phi of leading GenJets (MC5)",72,-M_PI, M_PI);
  m2jMC5gen = TH1F("m2jMC5gen","Dijet Mass of leading GenJets (MC5)",100,0.0,2*PtHistMax);

  //IC5 cal
  ptIC5cal  = TH1F("ptIC5cal","p_{T} of leading CaloJets (IC5)",50,0.0,PtHistMax);
  etaIC5cal = TH1F("etaIC5cal","#eta of leading CaloJets (IC5)",100,-5.0,5.0);
  phiIC5cal = TH1F("phiIC5cal","#phi of leading CaloJets (IC5)",72,-M_PI, M_PI);
  m2jIC5cal = TH1F("m2jIC5cal","Dijet Mass of leading CaloJets (IC5)",100,0.0,2*PtHistMax);

  //IC5 gen
  ptIC5gen = TH1F("ptIC5gen","p_{T} of leading GenJets (IC5)",50,0.0,PtHistMax);
  etaIC5gen = TH1F("etaIC5gen","#eta of leading GenJets (IC5)",100,-5.0,5.0);
  phiIC5gen = TH1F("phiIC5gen","#phi of leading GenJets (IC5)",72,-M_PI, M_PI);
  m2jIC5gen = TH1F("m2jIC5gen","Dijet Mass of leading GenJets (IC5)",100,0.0,2*PtHistMax);

  //KT10 cal
  ptKT10cal  = TH1F("ptKT10cal","p_{T} of leading CaloJets (KT10)",50,0.0,PtHistMax);
  etaKT10cal = TH1F("etaKT10cal","#eta of leading CaloJets (KT10)",100,-5.0,5.0);
  phiKT10cal = TH1F("phiKT10cal","#phi of leading CaloJets (KT10)",72,-M_PI, M_PI);
  m2jKT10cal = TH1F("m2jKT10cal","Dijet Mass of leading CaloJets (KT10)",100,0.0,2*PtHistMax);

  //KT10 gen
  ptKT10gen = TH1F("ptKT10gen","p_{T} of leading GenJets (KT10)",50,0.0,PtHistMax);
  etaKT10gen = TH1F("etaKT10gen","#eta of leading GenJets (KT10)",100,-5.0,5.0);
  phiKT10gen = TH1F("phiKT10gen","#phi of leading GenJets (KT10)",72,-M_PI, M_PI);
  m2jKT10gen = TH1F("m2jKT10gen","Dijet Mass of leading GenJets (KT10)",100,0.0,2*PtHistMax);

}

void JetValidation::analyze( const Event& evt, const EventSetup& es ) {
  math::XYZTLorentzVector p4jet[2];
  int jetInd;
  Handle<CaloJetCollection> caloJets;
  Handle<GenJetCollection> genJets;

  //Fill Basic Hists
  
  //MC5 cal
  evt.getByLabel( "midPointCone5CaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    ptMC5cal.Fill( cal->pt() );   
    etaMC5cal.Fill( cal->eta() );
    phiMC5cal.Fill( cal->phi() );
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2)m2jMC5cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 

  //MC5 gen
  evt.getByLabel( "midPointCone5GenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    ptMC5gen.Fill( gen->pt() );   
    etaMC5gen.Fill( gen->eta() );
    phiMC5gen.Fill( gen->phi() );
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2)m2jMC5gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 

  //IC5 cal
  evt.getByLabel( "iterativeCone5CaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    ptIC5cal.Fill( cal->pt() );   
    etaIC5cal.Fill( cal->eta() );
    phiIC5cal.Fill( cal->phi() );
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2)m2jIC5cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 

  //IC5 gen
  evt.getByLabel( "iterativeCone5GenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    ptIC5gen.Fill( gen->pt() );   
    etaIC5gen.Fill( gen->eta() );
    phiIC5gen.Fill( gen->phi() );
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2)m2jIC5gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 

  //KT10 cal
  evt.getByLabel( "ktCaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    ptKT10cal.Fill( cal->pt() );   
    etaKT10cal.Fill( cal->eta() );
    phiKT10cal.Fill( cal->phi() );
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2)m2jKT10cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 

  //KT10 gen
  evt.getByLabel( "ktGenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    ptKT10gen.Fill( gen->pt() );   
    etaKT10gen.Fill( gen->eta() );
    phiKT10gen.Fill( gen->phi() );
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2)m2jKT10gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 

}

void JetValidation::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
