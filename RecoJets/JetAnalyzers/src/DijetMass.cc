// DijetMass.cc
// Description:  Some Basic validation plots for jets.
// Author: Robert M. Harris
// Date:  30 - August - 2006
// Kalanand Mishra (November 22, 2009): 
//          Modified and cleaned up to work in 3.3.X
// 
#include "RecoJets/JetAnalyzers/interface/DijetMass.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
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


template<class Jet>
DijetMass<Jet>::DijetMass( const edm::ParameterSet & cfg ) {
  PtHistMax =  cfg.getUntrackedParameter<double>( "PtHistMax", 3000.0);
  EtaMax    = cfg.getUntrackedParameter<double> ("EtaMax", 1.3);  
  histogramFile    = cfg.getUntrackedParameter<std::string>("HistoFileName", 
							    "DijetMassHistos.root");

  AKJets =  cfg.getParameter<std::string> ("AKJets"); 
  AKCorJets =  cfg.getParameter<std::string> ("AKCorrectedJets"); 
  ICJets =  cfg.getParameter<std::string> ("ICJets"); 
  ICCorJets =  cfg.getParameter<std::string> ("ICCorrectedJets"); 
  SCJets =  cfg.getParameter<std::string> ("SCJets"); 
  SCCorJets =  cfg.getParameter<std::string> ("SCCorrectedJets"); 
  KTJets =  cfg.getParameter<std::string> ("KTJets"); 
  KTCorJets =  cfg.getParameter<std::string> ("KTCorrectedJets"); 
}

template<class Jet>
void DijetMass<Jet>::beginJob(  ) {
  cout << "DijetMass: Maximum bin edge for Pt Hists = " << PtHistMax << endl;
  numJets=2;

  //Initialize some stuff
  evtCount = 0;

  // Open the histogram file and book some associated histograms
  m_file=new TFile( histogramFile.c_str(),"RECREATE" ); 
  
  //Simple histos
  
  //AK unc
  ptAKunc  = TH1F("ptAKunc","p_{T} of leading Jets (AK)",50,0.0,PtHistMax);
  etaAKunc = TH1F("etaAKunc","#eta of leading Jets (AK)",23,-1.0,1.0);
  phiAKunc = TH1F("phiAKunc","#phi of leading Jets (AK)",72,-M_PI, M_PI);
  m2jAKunc = TH1F("m2jAKunc","Dijet Mass of leading Jets (AK)",100,0.0,2*PtHistMax);


  //AK cor
  ptAKcor  = TH1F("ptAKcor","p_{T} of leading Corrected Jets (AK)",50,0.0,PtHistMax);
  etaAKcor = TH1F("etaAKcor","#eta of leading Corrected Jets (AK)",23,-1.0,1.0);
  phiAKcor = TH1F("phiAKcor","#phi of leading Corrected Jets (AK)",72,-M_PI, M_PI);
  m2jAKcor = TH1F("m2jAKcor","Dijet Mass of leading Corrected Jets (AK)",100,0.0,2*PtHistMax);

  //IC unc
  ptICunc  = TH1F("ptICunc","p_{T} of leading Jets (IC)",50,0.0,PtHistMax);
  etaICunc = TH1F("etaICunc","#eta of leading Jets (IC)",23,-1.0,1.0);
  phiICunc = TH1F("phiICunc","#phi of leading Jets (IC)",72,-M_PI, M_PI);
  m2jICunc = TH1F("m2jICunc","Dijet Mass of leading Jets (IC)",100,0.0,2*PtHistMax);


  //IC cor
  ptICcor  = TH1F("ptICcor","p_{T} of leading Corrected Jets (IC)",50,0.0,PtHistMax);
  etaICcor = TH1F("etaICcor","#eta of leading Corrected Jets (IC)",23,-1.0,1.0);
  phiICcor = TH1F("phiICcor","#phi of leading Corrected Jets (IC)",72,-M_PI, M_PI);
  m2jICcor = TH1F("m2jICcor","Dijet Mass of leading Corrected Jets (IC)",100,0.0,2*PtHistMax);

  //KT unc
  ptKTunc  = TH1F("ptKTunc","p_{T} of leading Jets (KT)",50,0.0,PtHistMax);
  etaKTunc = TH1F("etaKTunc","#eta of leading Jets (KT)",23,-1.0,1.0);
  phiKTunc = TH1F("phiKTunc","#phi of leading Jets (KT)",72,-M_PI, M_PI);
  m2jKTunc = TH1F("m2jKTunc","Dijet Mass of leading Jets (KT)",100,0.0,2*PtHistMax);


  //KT cor
  ptKTcor  = TH1F("ptKTcor","p_{T} of leading Corrected Jets (KT)",50,0.0,PtHistMax);
  etaKTcor = TH1F("etaKTcor","#eta of leading Corrected Jets (KT)",23,-1.0,1.0);
  phiKTcor = TH1F("phiKTcor","#phi of leading Corrected Jets (KT)",72,-M_PI, M_PI);
  m2jKTcor = TH1F("m2jKTcor","Dijet Mass of leading Corrected Jets (KT)",100,0.0,2*PtHistMax);

  //SC unc
  ptSCunc  = TH1F("ptSCunc","p_{T} of leading Jets (SC)",50,0.0,PtHistMax);
  etaSCunc = TH1F("etaSCunc","#eta of leading Jets (SC)",23,-1.0,1.0);
  phiSCunc = TH1F("phiSCunc","#phi of leading Jets (SC)",72,-M_PI, M_PI);
  m2jSCunc = TH1F("m2jSCunc","Dijet Mass of leading Jets (SC)",100,0.0,2*PtHistMax);

  //SC cor
  ptSCcor  = TH1F("ptSCcor","p_{T} of leading Corrected Jets (SC)",50,0.0,PtHistMax);
  etaSCcor = TH1F("etaSCcor","#eta of leading Corrected Jets (SC)",23,-1.0,1.0);
  phiSCcor = TH1F("phiSCcor","#phi of leading Corrected Jets (SC)",72,-M_PI, M_PI);
  m2jSCcor = TH1F("m2jSCcor","Dijet Mass of leading Corrected Jets (SC)",100,0.0,2*PtHistMax);

}



template<class Jet>
void DijetMass<Jet>::analyze( const Event& evt, const EventSetup& es ) {

  evtCount++;
  math::XYZTLorentzVector p4jet[2];
  int jetInd;
  Handle<JetCollection> Jets;


  //Fill Simple Histos
  typename JetCollection::const_iterator i_jet;

  //AK unc
  evt.getByLabel( AKJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++ i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jAKunc.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptAKunc.Fill( p4jet[0].Pt() ); ptAKunc.Fill( p4jet[1].Pt() );  
     etaAKunc.Fill( p4jet[0].eta() ); etaAKunc.Fill( p4jet[1].eta() );  
     phiAKunc.Fill( p4jet[0].phi() ); phiAKunc.Fill( p4jet[1].phi() );  
   }


  //AK corrected
  evt.getByLabel( AKCorJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++ i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jAKcor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptAKcor.Fill( p4jet[0].Pt() ); ptAKcor.Fill( p4jet[1].Pt() );  
     etaAKcor.Fill( p4jet[0].eta() ); etaAKcor.Fill( p4jet[1].eta() );  
     phiAKcor.Fill( p4jet[0].phi() ); phiAKcor.Fill( p4jet[1].phi() );  
   }


  //IC unc
  evt.getByLabel( ICJets, Jets );
  jetInd = 0;
  for(  i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++ i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jICunc.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptICunc.Fill( p4jet[0].Pt() ); ptICunc.Fill( p4jet[1].Pt() );  
     etaICunc.Fill( p4jet[0].eta() ); etaICunc.Fill( p4jet[1].eta() );  
     phiICunc.Fill( p4jet[0].phi() ); phiICunc.Fill( p4jet[1].phi() );  
   }


  //IC corrected
  evt.getByLabel( ICCorJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++ i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jICcor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptICcor.Fill( p4jet[0].Pt() ); ptICcor.Fill( p4jet[1].Pt() );  
     etaICcor.Fill( p4jet[0].eta() ); etaICcor.Fill( p4jet[1].eta() );  
     phiICcor.Fill( p4jet[0].phi() ); phiICcor.Fill( p4jet[1].phi() );  
   }

  //KT unc
  evt.getByLabel( KTJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++ i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jKTunc.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptKTunc.Fill( p4jet[0].Pt() ); ptKTunc.Fill( p4jet[1].Pt() );  
     etaKTunc.Fill( p4jet[0].eta() ); etaKTunc.Fill( p4jet[1].eta() );  
     phiKTunc.Fill( p4jet[0].phi() ); phiKTunc.Fill( p4jet[1].phi() );  
   }


  //KT corrected
  evt.getByLabel( KTCorJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jKTcor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptKTcor.Fill( p4jet[0].Pt() ); ptKTcor.Fill( p4jet[1].Pt() );  
     etaKTcor.Fill( p4jet[0].eta() ); etaKTcor.Fill( p4jet[1].eta() );  
     phiKTcor.Fill( p4jet[0].phi() ); phiKTcor.Fill( p4jet[1].phi() );  
   }


  //SC unc
  evt.getByLabel( SCJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++i_jet  ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jSCunc.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptSCunc.Fill( p4jet[0].Pt() ); ptSCunc.Fill( p4jet[1].Pt() );  
     etaSCunc.Fill( p4jet[0].eta() ); etaSCunc.Fill( p4jet[1].eta() );  
     phiSCunc.Fill( p4jet[0].phi() ); phiSCunc.Fill( p4jet[1].phi() );  
   }


  //SC corrected
  evt.getByLabel( SCCorJets, Jets );
  jetInd = 0;
  for( i_jet = Jets->begin(); i_jet != Jets->end() && jetInd<2; ++i_jet ) {
    p4jet[jetInd] = i_jet->p4();
    jetInd++;
  }
  if(jetInd==2&&abs(p4jet[0].eta())<EtaMax &&abs(p4jet[1].eta())<EtaMax ){
     m2jSCcor.Fill( (p4jet[0]+p4jet[1]).mass() ); 
     ptSCcor.Fill( p4jet[0].Pt() ); ptSCcor.Fill( p4jet[1].Pt() );  
     etaSCcor.Fill( p4jet[0].eta() ); etaSCcor.Fill( p4jet[1].eta() );  
     phiSCcor.Fill( p4jet[0].phi() ); phiSCcor.Fill( p4jet[1].phi() );  
   }

}

template<class Jet>
void DijetMass<Jet>::endJob() {

  //Write out the histogram file.
  m_file->cd(); 

  ptAKunc.Write();
  etaAKunc.Write();
  phiAKunc.Write();
  m2jAKunc.Write();

  ptAKcor.Write();
  etaAKcor.Write();
  phiAKcor.Write();
  m2jAKcor.Write();

  ptICunc.Write();
  etaICunc.Write();
  phiICunc.Write();
  m2jICunc.Write();


  ptICcor.Write();
  etaICcor.Write();
  phiICcor.Write();
  m2jICcor.Write();

  ptKTunc.Write();
  etaKTunc.Write();
  phiKTunc.Write();
  m2jKTunc.Write();


  ptKTcor.Write();
  etaKTcor.Write();
  phiKTcor.Write();
  m2jKTcor.Write();

  ptSCunc.Write();
  etaSCunc.Write();
  phiSCunc.Write();
  m2jSCunc.Write();

  ptSCcor.Write();
  etaSCcor.Write();
  phiSCcor.Write();
  m2jSCcor.Write();

  m_file->Close(); 
}
#include "FWCore/Framework/interface/MakerMacros.h"
/////////// Calo Jet Instance ////////
typedef DijetMass<CaloJet> DijetMassCaloJets;
DEFINE_FWK_MODULE(DijetMassCaloJets);
/////////// Gen Jet Instance ////////
typedef DijetMass<GenJet> DijetMassGenJets;
DEFINE_FWK_MODULE(DijetMassGenJets);
/////////// PF Jet Instance ////////
typedef DijetMass<PFJet> DijetMassPFJets;
DEFINE_FWK_MODULE(DijetMassPFJets);
