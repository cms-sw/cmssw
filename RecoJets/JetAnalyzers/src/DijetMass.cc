// DijetMass.cc
// Description:  Some Basic validation plots for jets.
// Author: Robert M. Harris
// Date:  30 - August - 2006
// 
#include "RecoJets/JetAnalyzers/interface/DijetMass.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Math/interface/deltaR.h"
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
DijetMass::DijetMass( const ParameterSet & cfg ) :
  PtHistMax( cfg.getParameter<double>( "PtHistMax" ) ),
  GenType( cfg.getParameter<string>( "GenType" ) )
  {
}

void DijetMass::beginJob( const EventSetup & ) {
  cout << "DijetMass: Maximum bin edge for Pt Hists = " << PtHistMax << endl;
  numJets=2;

  //Initialize some stuff
  evtCount = 0;

  // Open the histogram file and book some associated histograms
  m_file=new TFile("DijetMassHistos.root","RECREATE"); 
  
  //Simple histos
  
  //MC5 cal
  ptMC5cal  = TH1F("ptMC5cal","p_{T} of leading CaloJets (MC5)",50,0.0,PtHistMax);
  etaMC5cal = TH1F("etaMC5cal","#eta of leading CaloJets (MC5)",23,-1.0,1.0);
  phiMC5cal = TH1F("phiMC5cal","#phi of leading CaloJets (MC5)",72,-M_PI, M_PI);
  m2jMC5cal = TH1F("m2jMC5cal","Dijet Mass of leading CaloJets (MC5)",100,0.0,2*PtHistMax);

  //MC5 gen
  ptMC5gen = TH1F("ptMC5gen","p_{T} of leading genJets (MC5)",50,0.0,PtHistMax);
  etaMC5gen = TH1F("etaMC5gen","#eta of leading genJets (MC5)",23,-1.0,1.0);
  phiMC5gen = TH1F("phiMC5gen","#phi of leading genJets (MC5)",72,-M_PI, M_PI);
  m2jMC5gen = TH1F("m2jMC5gen","Dijet Mass of leading genJets (MC5)",100,0.0,2*PtHistMax);

  //MC5 cor
  ptMC5cor  = TH1F("ptMC5cor","p_{T} of leading Corrected CaloJets (MC5)",50,0.0,PtHistMax);
  etaMC5cor = TH1F("etaMC5cor","#eta of leading Corrected CaloJets (MC5)",23,-1.0,1.0);
  phiMC5cor = TH1F("phiMC5cor","#phi of leading Corrected CaloJets (MC5)",72,-M_PI, M_PI);
  m2jMC5cor = TH1F("m2jMC5cor","Dijet Mass of leading Corrected CaloJets (MC5)",100,0.0,2*PtHistMax);

  //IC5 cal
  ptIC5cal  = TH1F("ptIC5cal","p_{T} of leading CaloJets (IC5)",50,0.0,PtHistMax);
  etaIC5cal = TH1F("etaIC5cal","#eta of leading CaloJets (IC5)",23,-1.0,1.0);
  phiIC5cal = TH1F("phiIC5cal","#phi of leading CaloJets (IC5)",72,-M_PI, M_PI);
  m2jIC5cal = TH1F("m2jIC5cal","Dijet Mass of leading CaloJets (IC5)",100,0.0,2*PtHistMax);

  //IC5 gen
  ptIC5gen = TH1F("ptIC5gen","p_{T} of leading genJets (IC5)",50,0.0,PtHistMax);
  etaIC5gen = TH1F("etaIC5gen","#eta of leading genJets (IC5)",23,-1.0,1.0);
  phiIC5gen = TH1F("phiIC5gen","#phi of leading genJets (IC5)",72,-M_PI, M_PI);
  m2jIC5gen = TH1F("m2jIC5gen","Dijet Mass of leading genJets (IC5)",100,0.0,2*PtHistMax);

  //IC5 cor
  ptIC5cor  = TH1F("ptIC5cor","p_{T} of leading Corrected CaloJets (IC5)",50,0.0,PtHistMax);
  etaIC5cor = TH1F("etaIC5cor","#eta of leading Corrected CaloJets (IC5)",23,-1.0,1.0);
  phiIC5cor = TH1F("phiIC5cor","#phi of leading Corrected CaloJets (IC5)",72,-M_PI, M_PI);
  m2jIC5cor = TH1F("m2jIC5cor","Dijet Mass of leading Corrected CaloJets (IC5)",100,0.0,2*PtHistMax);

  //KT10 cal
  ptKT10cal  = TH1F("ptKT10cal","p_{T} of leading CaloJets (KT10)",50,0.0,PtHistMax);
  etaKT10cal = TH1F("etaKT10cal","#eta of leading CaloJets (KT10)",23,-1.0,1.0);
  phiKT10cal = TH1F("phiKT10cal","#phi of leading CaloJets (KT10)",72,-M_PI, M_PI);
  m2jKT10cal = TH1F("m2jKT10cal","Dijet Mass of leading CaloJets (KT10)",100,0.0,2*PtHistMax);

  //KT10 gen
  ptKT10gen = TH1F("ptKT10gen","p_{T} of leading genJets (KT10)",50,0.0,PtHistMax);
  etaKT10gen = TH1F("etaKT10gen","#eta of leading genJets (KT10)",23,-1.0,1.0);
  phiKT10gen = TH1F("phiKT10gen","#phi of leading genJets (KT10)",72,-M_PI, M_PI);
  m2jKT10gen = TH1F("m2jKT10gen","Dijet Mass of leading genJets (KT10)",100,0.0,2*PtHistMax);


  //Matched jets Analysis Histograms for MC5 CaloJets only
  dR = TH1F("dR","Leading genJets dR with matched CaloJet",100,0,0.5);
  respVsPt = TProfile("respVsPt","CaloJet Response of Leading genJets in Barrel",100,0.0,PtHistMax/2); 
  dRcor = TH1F("dRcor","CorJets dR with matched CaloJet",100,0.0,0.01);
  corRespVsPt = TProfile("corRespVsPt","Corrected CaloJet Response of Leading genJets in Barrel",100,0.0,PtHistMax/2); 
}

void DijetMass::analyze( const Event& evt, const EventSetup& es ) {

  evtCount++;
  math::XYZTLorentzVector p4jet[2], p4gen[2], p4cal[2], p4cor[2];
  int jetInd;
  Handle<CaloJetCollection> caloJets;
  Handle<GenJetCollection> genJets;

  //Fill Simple Histos
  
  //MC5 cal
  evt.getByLabel( "midPointCone5CaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jMC5cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptMC5cal.Fill( p4jet[0].Pt() ); ptMC5cal.Fill( p4jet[1].Pt() );  
     etaMC5cal.Fill( p4jet[0].eta() ); etaMC5cal.Fill( p4jet[1].eta() );  
     phiMC5cal.Fill( p4jet[0].phi() ); phiMC5cal.Fill( p4jet[1].phi() );  
   }

  //MC5 gen
  evt.getByLabel( "midPointCone5GenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jMC5gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptMC5gen.Fill( p4jet[0].Pt() ); ptMC5gen.Fill( p4jet[1].Pt() );  
     etaMC5gen.Fill( p4jet[0].eta() ); etaMC5gen.Fill( p4jet[1].eta() );  
     phiMC5gen.Fill( p4jet[0].phi() ); phiMC5gen.Fill( p4jet[1].phi() );  
   }

  //MC5 cal
  evt.getByLabel( "corJetMcone5", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jMC5cor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptMC5cor.Fill( p4jet[0].Pt() ); ptMC5cor.Fill( p4jet[1].Pt() );  
     etaMC5cor.Fill( p4jet[0].eta() ); etaMC5cor.Fill( p4jet[1].eta() );  
     phiMC5cor.Fill( p4jet[0].phi() ); phiMC5cor.Fill( p4jet[1].phi() );  
   }


  //IC5 cal
  evt.getByLabel( "iterativeCone5CaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jIC5cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptIC5cal.Fill( p4jet[0].Pt() ); ptIC5cal.Fill( p4jet[1].Pt() );  
     etaIC5cal.Fill( p4jet[0].eta() ); etaIC5cal.Fill( p4jet[1].eta() );  
     phiIC5cal.Fill( p4jet[0].phi() ); phiIC5cal.Fill( p4jet[1].phi() );  
   }


  //IC5 gen
  evt.getByLabel( "iterativeCone5GenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jIC5gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptIC5gen.Fill( p4jet[0].Pt() ); ptIC5gen.Fill( p4jet[1].Pt() );  
     etaIC5gen.Fill( p4jet[0].eta() ); etaIC5gen.Fill( p4jet[1].eta() );  
     phiIC5gen.Fill( p4jet[0].phi() ); phiIC5gen.Fill( p4jet[1].phi() );  
   }

  //IC5 cor
  evt.getByLabel( "corJetIcone5", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jIC5cor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptIC5cor.Fill( p4jet[0].Pt() ); ptIC5cor.Fill( p4jet[1].Pt() );  
     etaIC5cor.Fill( p4jet[0].eta() ); etaIC5cor.Fill( p4jet[1].eta() );  
     phiIC5cor.Fill( p4jet[0].phi() ); phiIC5cor.Fill( p4jet[1].phi() );  
   }

  //KT10 cal
  evt.getByLabel( "ktCaloJets", caloJets );
  jetInd = 0;
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end() && jetInd<2; ++ cal ) {
    p4jet[jetInd] = cal->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jKT10cal.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptKT10cal.Fill( p4jet[0].Pt() ); ptKT10cal.Fill( p4jet[1].Pt() );  
     etaKT10cal.Fill( p4jet[0].eta() ); etaKT10cal.Fill( p4jet[1].eta() );  
     phiKT10cal.Fill( p4jet[0].phi() ); phiKT10cal.Fill( p4jet[1].phi() );  
   }

  //KT10 gen
  evt.getByLabel( "ktGenJets", genJets );
  jetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<2; ++ gen ) {
    p4jet[jetInd] = gen->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<1.0&&abs(p4jet[1].eta())<1.0){
     m2jKT10gen.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptKT10gen.Fill( p4jet[0].Pt() ); ptKT10gen.Fill( p4jet[1].Pt() );  
     etaKT10gen.Fill( p4jet[0].eta() ); etaKT10gen.Fill( p4jet[1].eta() );  
     phiKT10gen.Fill( p4jet[0].phi() ); phiKT10gen.Fill( p4jet[1].phi() );  
   }

  //Matching for MC5 Jets: leading genJets matched to any CaloJet
  evt.getByLabel( "midPointCone5GenJets", genJets );
  evt.getByLabel( "midPointCone5CaloJets", caloJets );
  jetInd = 0;
  double dRmin[2];
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd<numJets; ++ gen ) { //leading genJets
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
    if(dRmin[jetInd]>0.5)cout << "MC5 Match Warning: dR=" <<dRmin[jetInd]<<", GenPt="<<p4gen[jetInd].Pt()<<", CalPt="<<p4cal[jetInd].Pt()<<endl;
    jetInd++;    
  }
  //Fill Resp vs Pt profile histogram with response of two leading gen jets
  for( jetInd=0; jetInd<numJets; ++jetInd ){
    if(fabs(p4gen[jetInd].eta())<1.){
      respVsPt.Fill(p4gen[jetInd].Pt(), p4cal[jetInd].Pt()/p4gen[jetInd].Pt() );
    }
  }

  //Find the Corrected CaloJets that match the two uncorrected CaloJets
  evt.getByLabel( "corJetMcone5", caloJets );
  for( jetInd=0; jetInd<numJets; ++jetInd ){
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
  for( jetInd=0; jetInd<numJets; ++jetInd ){
    if(fabs(p4gen[jetInd].eta())<1.){
     corRespVsPt.Fill(p4gen[jetInd].Pt(), p4cor[jetInd].Pt()/p4gen[jetInd].Pt() ); 
    }
  }
   
}

void DijetMass::endJob() {

  //Write out the histogram file.
  m_file->Write(); 

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DijetMass);
