// JetValidation.cc
// Description:  Some Basic validation plots for jets.
// Author: Robert M. Harris
// Date:  30 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/JetValidation.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
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

  //Calorimeter Sub-System Analysis Histograms for IC5 CaloJets only
  emEnergyFraction =  TH1F("emEnergyFraction","Leading Jets EM Fraction",100,0.0,1.0);
  emEnergyInEB = TH1F("emEnergyInEB","Leading Jets emEnergyInEB",100,0.0,2*PtHistMax);
  emEnergyInEE = TH1F("emEnergyInEE","Leading Jets emEnergyInEE",100,0.0,2*PtHistMax);
  emEnergyInHF = TH1F("emEnergyInHF","Leading Jets emEnergyInHF",100,0.0,2*PtHistMax);
  hadEnergyInHB = TH1F("hadEnergyInHB","Leading Jets hadEnergyInHB",100,0.0,2*PtHistMax);
  hadEnergyInHE = TH1F("hadEnergyInHE","Leading Jets hadEnergyInHE",100,0.0,2*PtHistMax);
  hadEnergyInHF = TH1F("hadEnergyInHF","Leading Jets hadEnergyInHF",100,0.0,2*PtHistMax);
  hadEnergyInHO = TH1F("hadEnergyInHO","Leading Jets hadEnergyInHO",100,0.0,0.5*PtHistMax);
  
  //Matched jets Analysis Histograms for MC5 CaloJets only
  dR = TH1F("dR","Leading GenJets dR with matched CaloJet",100,0,0.5);
  respVsPt = TProfile("respVsPt","CaloJet Response of Leading GenJets in Barrel",100,0.0,PtHistMax/2); 
  dRcor = TH1F("dRcor","CorJets dR with matched CaloJet",100,0.0,0.01);
  corRespVsPt = TProfile("corRespVsPt","Corrected CaloJet Response of Leading GenJets in Barrel",100,0.0,PtHistMax/2); 

}

void JetValidation::analyze( const Event& evt, const EventSetup& es ) {
  math::XYZTLorentzVector p4jet[2], p4gen[2], p4cal[2], p4cor[2];
  int jetInd;
  Handle<CaloJetCollection> caloJets;
  Handle<GenJetCollection> genJets;

  //Fill Simple Histos
  
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

 //Calorimeter Sub-System Analysis Histograms for IC5 CaloJets only
  evt.getByLabel( "iterativeCone5CaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    emEnergyFraction.Fill(cal->emEnergyFraction()); 
    emEnergyInEB.Fill(cal->emEnergyInEB()); 
    emEnergyInEE.Fill(cal->emEnergyInEE()); 
    emEnergyInHF.Fill(cal->emEnergyInHF()); 
    hadEnergyInHB.Fill(cal->hadEnergyInHB()); 
    hadEnergyInHE.Fill(cal->hadEnergyInHE()); 
    hadEnergyInHF.Fill(cal->hadEnergyInHF()); 
    hadEnergyInHO.Fill(cal->hadEnergyInHO()); 
    jetInd++;
  }

  //Matching for MC5 Jets: leading genjets matched to any CaloJet
  evt.getByLabel( "midPointCone5GenJets", genJets );
  evt.getByLabel( "midPointCone5CaloJets", caloJets );
  jetInd = 0;
  double dRmin[2];
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) { //leading genJets
    p4gen[jetInd] = gen->p4();    //Gen 4-vector
    dRmin[jetInd]=1000.0;
    for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) { //all CaloJets
       double delR = deltaR( cal->eta(), cal->phi(), gen->eta(), gen->phi() ); 
       if(delR<dRmin[jetInd]){
         dRmin[jetInd]=delR;           //delta R of match
	 p4cal[jetInd] = cal->p4();  //Matched Cal 4-vector
       }
    }
    dR.Fill(dRmin[jetInd]);
    if(dRmin[jetInd]>0.5)cout << "Warning: dR=" <<dRmin<<", GenPt="<<p4gen[jetInd].Pt()<<", CalPt="<<p4cal[jetInd].Pt()<<endl;
    jetInd++;    
  }
  //Fill Resp vs Pt profile histogram with response of two leading gen jets
  for( jetInd=0; jetInd<2; ++jetInd ){
    if(fabs(p4gen[jetInd].eta())<1.){
      respVsPt.Fill(p4gen[jetInd].Pt(), p4cal[jetInd].Pt()/p4gen[jetInd].Pt() );
    }
  }

  //Find the Corrected CaloJets that match the two uncorrected CaloJets
  evt.getByLabel( "corJetMcone5", caloJets );
  for( jetInd=0; jetInd<2; ++jetInd ){
    bool found=kFALSE;
    for( CaloJetCollection::const_iterator cor = caloJets->begin(); cor != caloJets->end() && !found; ++ cor ) { //all corrected CaloJets
       double delR = deltaR( cor->eta(), cor->phi(), p4cal[jetInd].eta(),  p4cal[jetInd].phi()); 
       if(delR<0.01){
         dRmin[jetInd]=delR;           //delta R of match
	 p4cor[jetInd] = cor->p4();  //Matched Cal 4-vector
	 found=kTRUE;
         dRcor.Fill(dRmin[jetInd]);
       }
    }
    if(!found)cout << "Warning: corrected jet not found. jetInd=" << jetInd << endl;
  }
  //Fill Resp vs Pt profile histogram with corrected response of two leading gen jets
  for( jetInd=0; jetInd<2; ++jetInd ){
    if(fabs(p4gen[jetInd].eta())<1.){
     corRespVsPt.Fill(p4gen[jetInd].Pt(), p4cor[jetInd].Pt()/p4gen[jetInd].Pt() ); 
    }
  }
   
}

void JetValidation::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
