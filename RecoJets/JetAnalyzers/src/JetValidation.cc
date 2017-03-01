// JetValidation.cc
// Description:  Some Basic validation plots for jets.
// Author: K. Kousouris
// Date:  27 - August - 2008
// 
#include "RecoJets/JetAnalyzers/interface/JetValidation.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
JetValidation::JetValidation(edm::ParameterSet const& cfg)
{
  dRmatch             = cfg.getParameter<double> ("dRmatch");
  PtMin               = cfg.getParameter<double> ("PtMin");
  Njets               = cfg.getParameter<int> ("Njets");
  MCarlo              = cfg.getParameter<bool> ("MCarlo");
  genAlgo             = cfg.getParameter<string> ("genAlgo");
  calAlgo             = cfg.getParameter<string> ("calAlgo");
  jetTracksAssociator = cfg.getParameter<string> ("jetTracksAssociator"); 
  histoFileName       = cfg.getParameter<string> ("histoFileName");
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::beginJob() 
{
  m_file = new TFile(histoFileName.c_str(),"RECREATE"); 
  
  m_HistNames1D["CaloJetMulti"] = new TH1F("CaloJetMulti","Multiplicity of CaloJets",100,0,100);
  m_HistNames1D["ptCalo"] = new TH1F("ptCalo","p_{T} of CaloJets",7000,0,7000);
  m_HistNames1D["etaCalo"] = new TH1F("etaCalo","#eta of CaloJets",100,-5.0,5.0);
  m_HistNames1D["phiCalo"] = new TH1F("phiCalo","#phi of CaloJets",72,-M_PI, M_PI);
  m_HistNames1D["m2jCalo"] = new TH1F("m2jCalo","Dijet Mass of leading CaloJets",7000,0,14000);
  m_HistNames1D["nTracks"] = new TH1F("nTracks","Number of tracks associated with a jet",100,0,100);
  m_HistNames1D["chargeFraction"] = new TH1F("chargeFraction","Fraction of charged tracks pt",500,0,5);
  m_HistNames1D["emEnergyFraction"] = new TH1F("emEnergyFraction","Jets EM Fraction",110,0,1.1);
  m_HistNames1D["emEnergyInEB"] = new TH1F("emEnergyInEB","Jets emEnergyInEB",7000,0,14000);
  m_HistNames1D["emEnergyInEE"] = new TH1F("emEnergyInEE","Jets emEnergyInEE",7000,0,14000);
  m_HistNames1D["emEnergyInHF"] = new TH1F("emEnergyInHF","Jets emEnergyInHF",7000,0,14000);
  m_HistNames1D["hadEnergyInHB"] = new TH1F("hadEnergyInHB","Jets hadEnergyInHB",7000,0,14000);
  m_HistNames1D["hadEnergyInHE"] = new TH1F("hadEnergyInHE","Jets hadEnergyInHE",7000,0,14000);
  m_HistNames1D["hadEnergyInHF"] = new TH1F("hadEnergyInHF","Jets hadEnergyInHF",7000,0,14000);
  m_HistNames1D["hadEnergyInHO"] = new TH1F("hadEnergyInHO","Jets hadEnergyInHO",7000,0,14000);
  m_HistNamesProfile["EBfractionVsEta"] = new TProfile("EBfractionVsEta","Jets EBfraction vs #eta",100,-5.0,5.0);
  m_HistNamesProfile["EEfractionVsEta"] = new TProfile("EEfractionVsEta","Jets EEfraction vs #eta",100,-5.0,5.0);
  m_HistNamesProfile["HBfractionVsEta"] = new TProfile("HBfractionVsEta","Jets HBfraction vs #eta",100,-5.0,5.0);
  m_HistNamesProfile["HOfractionVsEta"] = new TProfile("HOfractionVsEta","Jets HOfraction vs #eta",100,-5.0,5.0);
  m_HistNamesProfile["HEfractionVsEta"] = new TProfile("HEfractionVsEta","Jets HEfraction vs #eta",100,-5.0,5.0);
  m_HistNamesProfile["HFfractionVsEta"] = new TProfile("HFfractionVsEta","Jets HFfraction vs #eta",100,-5.0,5.0); 
  m_HistNamesProfile["CaloEnergyVsEta"] = new TProfile("CaloEnergyVsEta","CaloJets Energy Vs. Eta",100,-5.0,5.0);
  m_HistNamesProfile["emEnergyVsEta"] = new TProfile("emEnergyVsEta","Jets EM Energy Vs. Eta",100,-5.0,5.0);
  m_HistNamesProfile["hadEnergyVsEta"] = new TProfile("hadEnergyVsEta","Jets HAD Energy Vs. Eta",100,-5.0,5.0);
  if (MCarlo)
    {
      m_HistNames1D["GenJetMulti"] = new TH1F("GenJetMulti","Multiplicity of GenJets",100,0,100);
      m_HistNames1D["ptHat"] = new TH1F("ptHat","p_{T}hat",7000,0,7000);
      m_HistNames1D["ptGen"] = new TH1F("ptGen","p_{T} of GenJets",7000,0,7000);
      m_HistNames1D["etaGen"] = new TH1F("etaGen","#eta of GenJets",100,-5.0,5.0);
      m_HistNames1D["phiGen"] = new TH1F("phiGen","#phi of GenJets",72,-M_PI, M_PI);
      m_HistNames1D["m2jGen"] = new TH1F("m2jGen","Dijet Mass of leading GenJets",7000,0,14000);
      m_HistNames1D["dR"] = new TH1F("dR","GenJets dR with matched CaloJet",200,0,1);
      m_HistNamesProfile["GenEnergyVsEta"] = new TProfile("GenEnergyVsEta","GenJets Energy Vs. Eta",100,-5.0,5.0);
      m_HistNamesProfile["respVsPtBarrel"] = new TProfile("respVsPtBarrel","CaloJet Response of GenJets in Barrel",7000,0,7000);
      m_HistNamesProfile["CaloErespVsEta"] = new TProfile("CaloErespVsEta","Jets Energy Response Vs. Eta",100,-5.0,5.0);
      m_HistNamesProfile["emErespVsEta"] = new TProfile("emErespVsEta","Jets EM Energy Response Vs. Eta",100,-5.0,5.0);
      m_HistNamesProfile["hadErespVsEta"] = new TProfile("hadErespVsEta","Jets HAD Energy Response Vs. Eta",100,-5.0,5.0);
    } 
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::analyze(edm::Event const& evt, edm::EventSetup const& iSetup) 
{
  math::XYZTLorentzVector p4jet[2];
  int jetInd,jetCounter,nTracks;
  double dRmin,dR,e,eta,emEB,emEE,emHF,hadHB,hadHE,hadHO,hadHF,pt,phi,pthat,chf;
  Handle<CaloJetCollection> caljets;
  Handle<GenJetCollection> genjets;
  Handle<double> genEventScale;
  Handle<JetTracksAssociation::Container> jetTracks;
  CaloJetCollection::const_iterator i_caljet;
  GenJetCollection::const_iterator i_genjet;
  evt.getByLabel(calAlgo,caljets);
  evt.getByLabel(jetTracksAssociator,jetTracks);
  jetInd = 0;
  jetCounter = 0;
  if (caljets->size()==0)
    cout<<"WARNING: NO calo jets in event "<<evt.id().event()<<", Run "<<evt.id().run()<<" !!!!"<<endl;
  for(i_caljet = caljets->begin(); i_caljet != caljets->end() && jetInd<Njets; ++i_caljet) 
    {
      e = i_caljet->energy();
      pt = i_caljet->pt();
      phi = i_caljet->phi(); 
      eta = i_caljet->eta();
      emEB = i_caljet->emEnergyInEB();
      emEE = i_caljet->emEnergyInEE();
      emHF = i_caljet->emEnergyInHF();
      hadHB = i_caljet->hadEnergyInHB();
      hadHE = i_caljet->hadEnergyInHE(); 
      hadHO = i_caljet->hadEnergyInHO(); 
      hadHF = i_caljet->hadEnergyInHF();  
      nTracks = JetTracksAssociation::tracksNumber(*jetTracks,*i_caljet);
      chf = (JetTracksAssociation::tracksP4(*jetTracks,*i_caljet)).pt()/pt;
      if (jetInd<2)
        p4jet[jetInd] = i_caljet->p4();
      if (pt>PtMin)
        {  
          FillHist1D("ptCalo",pt);
          FillHist1D("etaCalo",eta);
          FillHist1D("phiCalo",phi);
          FillHist1D("emEnergyFraction",i_caljet->emEnergyFraction());
	  FillHist1D("nTracks",nTracks);
	  FillHist1D("chargeFraction",chf); 
          FillHist1D("emEnergyInEB",emEB); 
          FillHist1D("emEnergyInEE",emEE); 
          FillHist1D("emEnergyInHF",emHF); 
          FillHist1D("hadEnergyInHB",hadHB); 
          FillHist1D("hadEnergyInHE",hadHE); 
          FillHist1D("hadEnergyInHF",hadHF); 
          FillHist1D("hadEnergyInHO",hadHO);
          FillHistProfile("EBfractionVsEta",eta,emEB/e);
          FillHistProfile("EEfractionVsEta",eta,emEE/e);
          FillHistProfile("HBfractionVsEta",eta,hadHB/e);
          FillHistProfile("HOfractionVsEta",eta,hadHO/e);
          FillHistProfile("HEfractionVsEta",eta,hadHE/e);
          FillHistProfile("HFfractionVsEta",eta,(hadHF+emHF)/e);
          FillHistProfile("CaloEnergyVsEta",eta,e);
          FillHistProfile("emEnergyVsEta",eta,emEB+emEE+emHF);
          FillHistProfile("hadEnergyVsEta",eta,hadHB+hadHO+hadHE+hadHF);
          jetCounter++;
        }
      jetInd++;
    }
  FillHist1D("CaloJetMulti",jetCounter);
  if (jetInd>1)
    FillHist1D("m2jCalo",(p4jet[0]+p4jet[1]).mass());
  if (MCarlo)
    { 
      evt.getByLabel(genAlgo,genjets);  
      evt.getByLabel("genEventScale",genEventScale);
      pthat = *genEventScale;
      FillHist1D("ptHat",pthat);
      CaloJet MatchedJet;
      jetInd = 0;
      if (genjets->size()==0)
        cout<<"WARNING: NO gen jets in event "<<evt.id().event()<<", Run "<<evt.id().run()<<" !!!!"<<endl;
      for(i_genjet = genjets->begin(); i_genjet != genjets->end() && jetInd<Njets; ++i_genjet) 
        {
          if (jetInd<2)
            p4jet[jetInd] = i_genjet->p4();
          FillHist1D("ptGen",i_genjet->pt());
          FillHist1D("etaGen",i_genjet->eta());
          FillHist1D("phiGen",i_genjet->phi());
          FillHistProfile("GenEnergyVsEta",i_genjet->eta(),i_genjet->energy());
          dRmin=1000.0;
          for(i_caljet = caljets->begin(); i_caljet != caljets->end(); ++i_caljet)
            {
              dR = deltaR(i_caljet->eta(),i_caljet->phi(),i_genjet->eta(),i_genjet->phi());
              if (dR<dRmin)
                {
                  dRmin = dR;           
	          MatchedJet = *i_caljet;       
                }
            }
          FillHist1D("dR",dRmin); 
          e = MatchedJet.energy();
          pt = MatchedJet.pt();
          eta = MatchedJet.eta();
          emEB = MatchedJet.emEnergyInEB();
          emEE = MatchedJet.emEnergyInEE();
          emHF = MatchedJet.emEnergyInHF();
          hadHB = MatchedJet.hadEnergyInHB();
          hadHE = MatchedJet.hadEnergyInHE(); 
          hadHO = MatchedJet.hadEnergyInHO(); 
          hadHF = MatchedJet.hadEnergyInHF();    
          if (dRmin<dRmatch && pt>PtMin)
            {
              FillHistProfile("CaloErespVsEta",eta,e/i_genjet->energy());
              FillHistProfile("emErespVsEta",eta,(emEB+emEE+emHF)/i_genjet->energy());
              FillHistProfile("hadErespVsEta",eta,(hadHB+hadHO+hadHE+hadHF)/i_genjet->energy());
              if (fabs(i_genjet->eta())<1.)
                FillHistProfile("respVsPtBarrel",i_genjet->pt(),pt/i_genjet->pt()); 
            }
          jetInd++;
        } 
      FillHist1D("GenJetMulti",jetInd);
      if (jetInd>1)
        FillHist1D("m2jGen",(p4jet[0]+p4jet[1]).mass()); 
    }
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::endJob() 
{
  /////////// Write Histograms in output ROOT file ////////
  if (m_file !=0) 
    {
      m_file->cd();
      for (std::map<TString, TH1*>::iterator hid = m_HistNames1D.begin(); hid != m_HistNames1D.end(); hid++)
        hid->second->Write();
      for (std::map<TString, TH2*>::iterator hid = m_HistNames2D.begin(); hid != m_HistNames2D.end(); hid++)
        hid->second->Write();
      for (std::map<TString, TProfile*>::iterator hid = m_HistNamesProfile.begin(); hid != m_HistNamesProfile.end(); hid++)
        hid->second->Write(); 
      delete m_file;
      m_file = 0;      
    }
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::FillHist1D(const TString& histName,const Double_t& value) 
{
  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value);
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::FillHist2D(const TString& histName,const Double_t& valuex,const Double_t& valuey) 
{
  std::map<TString, TH2*>::iterator hid=m_HistNames2D.find(histName);
  if (hid==m_HistNames2D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(valuex,valuey);
}
////////////////////////////////////////////////////////////////////////////////////////
void JetValidation::FillHistProfile(const TString& histName,const Double_t& valuex,const Double_t& valuey) 
{
  std::map<TString, TProfile*>::iterator hid=m_HistNamesProfile.find(histName);
  if (hid==m_HistNamesProfile.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(valuex,valuey);
} 
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetValidation);
