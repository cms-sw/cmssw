// Name: JetAnaPythia
// Description:  Example of analysis of Pythia produced partons & jets
//               Based on Kostas Kousouris' templated JetPlotsExample.
//               Plots are tailored to needs of dijet mass and ratio analysis.
// Author: R. Harris
// Date:  28 - Oct - 2008
#include "RecoJets/JetAnalyzers/interface/JetAnaPythia.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TFile.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
JetAnaPythia<Jet>::JetAnaPythia(edm::ParameterSet const& cfg)
{
  JetAlgorithm  = cfg.getParameter<std::string> ("JetAlgorithm"); 
  HistoFileName = cfg.getParameter<std::string> ("HistoFileName");
  NJets         = cfg.getParameter<int> ("NJets");
  debug         = cfg.getParameter<bool> ("debug");
  eventsGen     = cfg.getParameter<int> ("eventsGen");
  anaLevel      = cfg.getParameter<std::string> ("anaLevel");
  xsecGen       = cfg.getParameter< vector<double> > ("xsecGen");
  ptHatEdges    = cfg.getParameter< vector<double> > ("ptHatEdges");

}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetAnaPythia<Jet>::beginJob() 
{
  TString hname;
  m_file = new TFile(HistoFileName.c_str(),"RECREATE"); 
  /////////// Booking histograms //////////////////////////
  const int nMassBins = 103;
  double massBoundaries[nMassBins+1] = {1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 176, 197, 220, 244, 270, 296, 325, 354, 386, 419, 453, 489, 526, 565, 606, 649, 693, 740, 788, 838, 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687, 1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 4509, 4686, 4869, 5058, 5253, 5455, 5663, 5877, 6099, 6328, 6564, 6808, 7060, 7320, 7589, 7866, 8152, 8447, 8752, 9067, 9391, 9726, 10072, 10430, 10798, 11179, 11571, 11977, 12395, 12827, 13272, 13732, 14000};  

  hname = "JetPt";
  m_HistNames1D[hname] = new TH1F(hname,hname,500,0,5000);

  hname = "JetEta";
  m_HistNames1D[hname] = new TH1F(hname,hname,120,-6,6);

  hname = "JetPhi";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,-M_PI,M_PI);

  hname = "NumberOfJets";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,0,100);

  hname = "DijetMass";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DijetMassWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "DijetMassIn";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DijetMassInWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "DijetMassOut";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DijetMassOutWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "ResonanceMass";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DipartonMass";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DipartonMassWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "DipartonMassIn";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DipartonMassInWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "DipartonMassOut";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);

  hname = "DipartonMassOutWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,nMassBins,massBoundaries);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "PtHat";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000,0,5000);

  hname = "PtHatWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,1000,0,5000);
  m_HistNames1D.find(hname)->second->Sumw2();

  hname = "PtHatFine";
  m_HistNames1D[hname] = new TH1F(hname,hname,5000,0,5000);

  hname = "PtHatFineWt";
  m_HistNames1D[hname] = new TH1F(hname,hname,5000,0,5000);
  m_HistNames1D.find(hname)->second->Sumw2();

  mcTruthTree_   = new TTree("mcTruthTree","mcTruthTree");
  mcTruthTree_->Branch("xsec",     &xsec,      "xsec/F");
  mcTruthTree_->Branch("weight",     &weight,      "weight/F");
  mcTruthTree_->Branch("pt_hat",     &pt_hat,      "pt_hat/F");
  mcTruthTree_->Branch("nJets",     &nJets,      "nJets/I");
  mcTruthTree_->Branch("etaJet1",     &etaJet1,      "etaJet1/F");
  mcTruthTree_->Branch("etaJet2",     &etaJet2,      "etaJet2/F");
  mcTruthTree_->Branch("ptJet1",     &ptJet1,      "ptJet1/F");
  mcTruthTree_->Branch("ptJet2",     &ptJet2,      "ptJet2/F");
  mcTruthTree_->Branch("diJetMass",     &diJetMass,      "diJetMass/F");
  mcTruthTree_->Branch("etaPart1",     &etaPart1,      "etaPart1/F");
  mcTruthTree_->Branch("etaPart2",     &etaPart2,      "etaPart2/F");
  mcTruthTree_->Branch("ptPart1",     &ptPart1,      "ptPart1/F");
  mcTruthTree_->Branch("ptPart2",     &ptPart2,      "ptPart2/F");
  mcTruthTree_->Branch("diPartMass",     &diPartMass,      "diPartMass/F");

  
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetAnaPythia<Jet>::analyze(edm::Event const& evt, edm::EventSetup const& iSetup) 
{
  int notDone=1;
  while(notDone)
  {  //while loop to allow us to tailor the analysis level for faster running.
    TString hname; 
      
    // Process Info
  
    //edm::Handle< double > genEventScale;
    //evt.getByLabel("genEventScale", genEventScale );
    //pt_hat = *genEventScale;

    edm::Handle<edm::HepMCProduct> MCevt;
    evt.getByLabel("generatorSmeared", MCevt);
    HepMC::GenEvent * myGenEvent = new HepMC::GenEvent(*(MCevt->GetEvent()));
   
    double pthat = myGenEvent->event_scale();
    pt_hat = float(pthat);

    delete myGenEvent;

    if(anaLevel != "generating"){  //We are not generating events, so xsec is there
      //edm::Handle< GenRunInfoProduct > genInfoProduct;
      ///evt.getRun().getByLabel("generator", genInfoProduct );
      //xsec = (double)genInfoProduct->externalXSecLO();
       xsec=0.0;
       if( ptHatEdges.size()>xsecGen.size() ){
         for(unsigned int i_pthat = 0; i_pthat < xsecGen.size(); ++i_pthat){
            if( pthat >= ptHatEdges[i_pthat] &&  pthat < ptHatEdges[i_pthat+1])xsec=float(xsecGen[i_pthat]);
         }
       }
       else 
       {
        std::cout << "Number of PtHat bin edges too small. Xsec set to zero" << std::endl;  
       }
    }
    else
    {                        
      xsec = xsecGen[0];   //Generating events, no xsec in event, get xsec from user input
    } 
    if(debug)std::cout << "cross section=" <<xsec << " pb" << std::endl;           
    weight =  xsec/eventsGen;

    if(debug)std::cout << "pt_hat=" <<pt_hat  <<  std::endl;
    hname = "PtHat";
    FillHist1D(hname, pt_hat, 1.0); 
    hname = "PtHatFine";
    FillHist1D(hname, pt_hat, 1.0); 
    hname = "PtHatWt";
    FillHist1D(hname, pt_hat, weight); 
    hname = "PtHatFineWt";
    FillHist1D(hname, pt_hat, weight); 
    if(anaLevel=="PtHatOnly")break;  //ptHatOnly should be very fast

    // Jet Info
    math::XYZTLorentzVector p4jet[2];
    float etajet[2];
    /////////// Get the jet collection //////////////////////
    Handle<JetCollection> jets;
    evt.getByLabel(JetAlgorithm,jets);
    typename JetCollection::const_iterator i_jet;
    int index = 0;

    /////////// Count the jets in the event /////////////////
    hname = "NumberOfJets";
    nJets = jets->size();
    FillHist1D(hname,nJets,1.0); 
    
  
    // Two Leading Jet Info
    for(i_jet = jets->begin(); i_jet != jets->end() && index < 2; ++i_jet) 
      {
        hname = "JetPt";
        FillHist1D(hname,i_jet->pt(),1.0);   
        hname = "JetEta";
        FillHist1D(hname,i_jet->eta(),1.0);
        hname = "JetPhi";
        FillHist1D(hname,i_jet->phi(),1.0);
        p4jet[index] = i_jet->p4();
        etajet[index] = i_jet->eta();
        if(debug)std::cout << "jet " << index+1 <<": pt=" <<i_jet->pt() << ", eta=" <<etajet[index] <<  std::endl;
        index++;
      }

      // TTree variables //
      etaJet1 = etajet[0];
      etaJet2 = etajet[1];
      ptJet1 = p4jet[0].pt();
      ptJet2 = p4jet[1].pt();
      diJetMass = (p4jet[0]+p4jet[1]).mass();

     ///  Histograms for Dijet Mass Analysis  ////
      if(index==2&&abs(etaJet1)<1.3&&abs(etaJet2)<1.3){
       hname = "DijetMass";
       FillHist1D(hname,diJetMass ,1.0); 
       hname = "DijetMassWt";
       FillHist1D(hname,diJetMass ,weight); 
     }

      /// Histograms for Dijet Ratio Analysis: Inner region ///
      if(index==2&&abs(etaJet1)<0.7&&abs(etaJet2)<0.7){
       hname = "DijetMassIn";
       FillHist1D(hname,diJetMass ,1.0); 
       hname = "DijetMassInWt";
       FillHist1D(hname,diJetMass ,weight); 
     }
      /// Histograms for Dijet Ratio Analysis: Outer region ////
      if(index==2 && (abs(etaJet1)>0.7&&abs(etaJet1)<1.3) 
                  && (abs(etaJet2)>0.7&&abs(etaJet2)<1.3) ){
       hname = "DijetMassOut";
       FillHist1D(hname, diJetMass ,1.0); 
       hname = "DijetMassOutWt";
       FillHist1D(hname,diJetMass ,weight); 
     }
     if(anaLevel=="Jets")break;  //Jets level for samples without genParticles
  

     // Parton Info
     edm::Handle<std::vector<reco::GenParticle> > genParticlesHandle_;
     evt.getByLabel("genParticles",genParticlesHandle_);
     if(debug)for( size_t i = 0; i < genParticlesHandle_->size(); ++ i ) {
       const reco::GenParticle & p = (*genParticlesHandle_)[i];
       int id = p.pdgId();
       int st = p.status();
       math::XYZTLorentzVector genP4 = p.p4();
       if(i>=2&&i<=8)std::cout << "particle " << i << ": id=" << id << ", status=" << st << ", mass=" << genP4.mass() << ", pt=" <<  genP4.pt() << ", eta=" << genP4.eta() << std::endl; 
     }
     // Examine the 7th particle in pythia.
     // It should be either a resonance (abs(id)>=32) or the first outgoing parton
     // for the processes we will consider: dijet resonances, QCD, or QCD +contact interactions.
     const reco::GenParticle & p = (*genParticlesHandle_)[6];
     int id = p.pdgId();
     math::XYZTLorentzVector resonance_p, parton1_p, parton2_p;
     if(abs(id)>=32){
        /// We are looking at dijet resonances. ////
        resonance_p = p.p4();      
        hname = "ResonanceMass";
        FillHist1D(hname,resonance_p.mass() ,1.0); 
        const reco::GenParticle & q = (*genParticlesHandle_)[7];
        parton1_p = q.p4();
        const reco::GenParticle & r = (*genParticlesHandle_)[8];
        parton2_p = r.p4();
        if(debug)std::cout << "Resonance mass=" << resonance_p.mass() << ", parton 1 pt=" << parton1_p.pt()  << ", parton 2 pt=" << parton2_p.pt() << ", diparton mass=" << (parton1_p+parton2_p).mass() << std::endl;
     }
     else
     {
        ///  We are looking at QCD   ////
        parton1_p = p.p4();
        const reco::GenParticle & q = (*genParticlesHandle_)[7];
        parton2_p = q.p4();
        if(debug)std::cout <<  "parton 1 pt=" << parton1_p.pt()  << ", parton 2 pt=" << parton2_p.pt() << ", diparton mass=" << (parton1_p+parton2_p).mass() << std::endl;
     }

      etaPart1 = parton1_p.eta();
      etaPart2 = parton2_p.eta();
      ptPart1 = parton1_p.pt();
      ptPart2 = parton2_p.pt();  
      diPartMass = (parton1_p+parton2_p).mass();  
     /// Diparton mass for dijet mass analysis  ////
     if(abs(etaPart1)<1.3&&abs(etaPart2)<1.3){
       hname = "DipartonMass";
       FillHist1D(hname,diPartMass ,1.0); 
       hname = "DipartonMassWt";
       FillHist1D(hname,diPartMass ,weight); 
     }
     /// Diparton mass for dijet ratio analysis: inner region ///
     if(abs(etaPart1)<0.7&&abs(etaPart2)<0.7){
       hname = "DipartonMassIn";
       FillHist1D(hname,diPartMass ,1.0); 
       hname = "DipartonMassInWt";
       FillHist1D(hname,diPartMass ,weight); 
     }
     /// Diparton mass for dijet ratio analysis: outer region ///
     if(    (abs(etaPart1)>0.7&&abs(etaPart1)<1.3)
         && (abs(etaPart2)>0.7&&abs(etaPart2)<1.3) ){
       hname = "DipartonMassOut";
       FillHist1D(hname,diPartMass ,1.0); 
       hname = "DipartonMassOutWt";
       FillHist1D(hname,diPartMass ,weight); 
     }

     // Fill the TTree //
     mcTruthTree_->Fill();
   
     notDone=0;  //We are done, exit the while loop
   }//end of while

}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetAnaPythia<Jet>::endJob() 
{
  /////////// Write Histograms in output ROOT file ////////
  if (m_file !=0) 
    {
      m_file->cd();
      mcTruthTree_->Write(); 
      for (std::map<TString, TH1*>::iterator hid = m_HistNames1D.begin(); hid != m_HistNames1D.end(); hid++)
        hid->second->Write();
      delete m_file;
      m_file = 0;      
    }
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetAnaPythia<Jet>::FillHist1D(const TString& histName,const Double_t& value, const Double_t& wt) 
{
  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value,wt);
}
/////////// Register Modules ////////
#include "FWCore/Framework/interface/MakerMacros.h"
/////////// Calo Jet Instance ////////
typedef JetAnaPythia<CaloJet> CaloJetAnaPythia;
DEFINE_FWK_MODULE(CaloJetAnaPythia);
/////////// Cen Jet Instance ////////
typedef JetAnaPythia<GenJet> GenJetAnaPythia;
DEFINE_FWK_MODULE(GenJetAnaPythia);
/////////// PF Jet Instance ////////
typedef JetAnaPythia<PFJet> PFJetAnaPythia;
DEFINE_FWK_MODULE(PFJetAnaPythia);
