#include <iostream>
#include <TH1.h>
#include <TFile.h>
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetfwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"
#include <vector>


using namespace std;


double Et(double E, double Pt, double Z)
{
  double et=0.0;
  if(Pt==0)
    return et;
  et=sqrt(E*E*(Pt*Pt)/(Pt*Pt+Z*Z));
  return et;
}

//void PFBenchmarkAlgo::setOutputRootFileName(string *outputRootFileName)
//{
//  this->outputRootFileName_=outputRootFileName;
//}

void PFBenchmarkAlgo::setCaloJets(edm::Handle<reco::CaloJetCollection> caloJets)
{
  this->caloJets_=caloJets;
}

void PFBenchmarkAlgo::setGenJets(edm::Handle<reco::GenJetCollection> genJets)
{
  this->genJets_=genJets;
}

void  PFBenchmarkAlgo::setPfJets(edm::Handle<reco::PFJetCollection> pfJets)
{
  this->pfJets_=pfJets;
}

void  PFBenchmarkAlgo::setHepMC(edm::Handle<edm::HepMCProduct> hepMC)
{
  this->hepMC_=hepMC;
}

PFBenchmarkAlgo::PFBenchmarkAlgo(string outputRootFileName)
{
  this->outputRootFileName_=outputRootFileName;
  file_ = TFile::Open(outputRootFileName_.c_str(), "RECREATE");
  // 1 Comparion RecJets with HepmMC 
  h_deltaETvisible_MCHEPMC_PF_
  = new TH1F("h_deltaETvisible_MCHEPMC_PF_" , "Jet Et difference PF - HepMC Truth;#Delta E_{T}"
  ,150,-150,150);//100,-100,100
  h_deltaETvisible_MCHEPMC_EHT_
  = new TH1F("h_deltaETvisible_MCHEPMC_EHT_" , "Jet Et difference EHT - HepMC Truth;#Delta E_{T}"
  ,150,-150,150);
 
  // A Delta Et / true
  h_deltaETDivTrue_MCHEPMC_PF_
  = new TH1F("h_deltaETDivTrue_MCHEPMC_PF_" , "#Delta E_{T} / E_{T true} - PF;#Delta E_{T}"
  ,150,-2,2);//100,-100,100
  h_deltaETDivTrue_MCHEPMC_EHT_
  = new TH1F("h_deltaETDivTrue_MCHEPMC_EHT_" , "#Delta E_{T} / E_{T true} - EHT;#Delta E_{T}"
  ,150,-2,2);

   // B Delta Et / reco
   h_deltaETDivReco_MCHEPMC_PF_
   = new TH1F("h_deltaETDivReco_MCHEPMC_PF_" , "Jet Et difference PF - HepMC Truth;#Delta E_{T}"
   ,150,-2,2);//100,-100,100
   h_deltaETDivReco_MCHEPMC_EHT_
   = new TH1F("h_deltaETDivReco_MCHEPMC_EHT_" , "Jet Et difference EHT - HepMC Truth;#Delta E_{T}"
   ,150,-2,2);


   // 2 Delta Et versus Pt
   h_deltaEtvsEt_MCHEPMC_PF_
   = new TH1F("h_deltaEtvsEt_MCHEPMC_PF_" , "#Delta E_{T} vs p_{T} - PF;p_{T};#Delta E_{T}"
   ,50,0,200); //50,0,55//
   h_deltaEtvsEt_MCHEPMC_EHT_
   = new TH1F("h_deltaEtvsEt_MCHEPMC_EHT_" , "#Delta E_{T} vs p_{T} - EHT;p_{T};#Delta E_{T}"
   ,50,0,200);
   n_deltaEtvsEt=new TH1F("n_deltaEtvsEt","qqq",50,0,200);

   // 3 Delta Et versus eta
   h_deltaEtvsEta_MCHEPMC_PF_
   = new TH1F("h_deltaEtvsEta_MCHEPMC_PF_" , "#Delta E_{T} vs #eta - PF;#eta;#Delta E_{T}"
   ,80,-3,3);
   h_deltaEtvsEta_MCHEPMC_EHT_
   = new TH1F("h_deltaEtvsEta_MCHEPMC_EHT_" , "#Delta E_{T} vs #eta - EHT;#eta;#Delta E_{T}"
   ,80,-3,3);
   n_deltaEtvsEta=new TH1F("n_deltaEtvsEta","qqq",80,-3,3);

   // 4 delta Et / Et(true) vs pT
   h_deltaEtDivEtvsEt_MCHEPMC_PF_
   = new TH1F("h_deltaEtDivEtvsEt_MCHEPMC_PF_" , "#Delta E_{T}/E_{T true} vs p_{T} - PF;p_{T};#Delta E_{T}/E_{T true}"
   ,50,0,200);//50,0,200
   h_deltaEtDivEtvsEt_MCHEPMC_EHT_
   = new TH1F("h_deltaEtDivEtvsEt_MCHEPMC_EHT_" , "#Delta E_{T}/E_{T true} vs p_{T} - EHT;p_{T};#Delta E_{T}/E_{T true}"
   ,50,0,200);
   n_deltaEtDivEtvsEt=new TH1F("n_deltaEtDivEtvsEt","qqq",50,0,200);

   // 5 delta Et / Et(rec) vs pT
   h_deltaEtDivEtrecvsEt_MCHEPMC_PF_
   = new TH1F("h_deltaEtDivEtrecvsEt_MCHEPMC_PF_" , "#Delta E_{T}/E_{T rec} vs p_{T} - PF;p_{T};#Delta E_{T}/E_{T}"
   ,50,0,200);
   h_deltaEtDivEtrecvsEt_MCHEPMC_EHT_
   = new TH1F("h_deltaEtDivEtrecvsEt_MCHEPMC_EHT_" , "#Delta E_{T}/E_{T rec} vs  p_{T} - EHT;p_{T};#Delta E_{T}/E_{T}"
   ,50,0,200);
   n_deltaEtDivEtrecvsEt=new TH1F("n_deltaEtDivEtrecvsEt","qqq",50,0,200);

   // 6 delta Et / Et(true) vs eta
   h_deltaEtDivEtvsEta_MCHEPMC_PF_
   = new TH1F("h_deltaEtDivEtvsEta_MCHEPMC_PF_" , "#Delta E_{T}/E_{T true} vs #eta - PF;#eta;#Delta E_{T}/E_{T true}"
   ,80,-3,3);
   h_deltaEtDivEtvsEta_MCHEPMC_EHT_
   = new TH1F("h_deltaEtDivEtvsEta_MCHEPMC_EHT_" , "#Delta E_{T}/E_{T true} vs #eta - EHT;#eta;#Delta E_{T}/E_{T true}"
   ,80,-3,3);
   n_deltaEtDivEtvsEta=new TH1F("n_deltaEtDivEtvsEta","qqq",80,-3,3);

   //7 delta Et / Et(rec) vs eta
   h_deltaEtDivEtrecvsEta_MCHEPMC_PF_
   = new TH1F("h_deltaEtDivEtrecvsEta_MCHEPMC_PF_" , "#Delta E_{T}/E_{T rec} vs #eta - PF;#eta;#Delta E_{T}/E_{T rec}"
   ,150,-3,3);
   h_deltaEtDivEtrecvsEta_MCHEPMC_EHT_
   = new TH1F("h_deltaEtDivEtrecvsEta_MCHEPMC_EHT_" , "#Delta E_{T}/E_{T rec} vs #eta - EHT;#eta;#Delta E_{T}/E_{T rec}"
   ,150,-3,3);
   n_deltaEtDivEtrecvsEta=new TH1F("n_deltaEtDivEtrecvsEta","qqq",80,-3,3);

	//  ------------- eta 

   // 8 delta eta
   h_deltaEta_MCHEPMC_PF_
   = new TH1F("h_deltaEta_MCHEPMC_PF_","#Delta #eta - PF;#Delta #eta"
   ,250,-0.2,0.2);
   h_deltaEta_MCHEPMC_EHT_
   = new TH1F("h_deltaEta_MCHEPMC_EHT_","#Delta #eta - EHT;#Delta #eta"
   ,250,-0.2,0.2);

   // 9 deltaEta vs pt
   h_deltaEtavsPt_MCHEPMC_PF_
   = new TH1F("h_deltaEtavsPt_MCHEPMC_PF_","#Delta #eta vs p_{T} - PF;p_{T};#Delta #eta"
   ,50,0,200);
   h_deltaEtavsPt_MCHEPMC_EHT_
   = new TH1F("h_deltaEtavsPt_MCHEPMC_EHT_","#Delta #eta vs p_{T} - EHT;p_{T};#Delta #eta"
   ,50,0,200);
   n_deltaEtavsPt=new TH1F("n_deltaEtavsPt","qqq",50,0,200);

   // 10 delta eta vs eta
   h_deltaEtavsEta_MCHEPMC_PF_
   = new TH1F("h_deltaEtavsEta_MCHEPMC_PF_","#Delta #eta vs #eta - PF;#eta;#Delta #eta"
   ,80,-3,3);
   h_deltaEtavsEta_MCHEPMC_EHT_
   = new TH1F("h_deltaEtavsEta_MCHEPMC_EHT_","#Delta #eta vs #eta - EHT;#eta;#Delta #eta"
   ,80,-3,3);
   n_deltaEtavsEta=new TH1F("n_deltaEtavsEta","qqq",80,-3,3);

   // C delta ets/ eta
   h_deltaEtaDivReco_MCHEPMC_PF_
   = new TH1F("h_deltaEtaDivReco_MCHEPMC_PF_" , "#Delta #eta / #eta_{reco} - PF;#Delta #eta / #eta_{reco}"
   ,150,-1,1);
   h_deltaEtaDivReco_MCHEPMC_EHT_
   = new TH1F("h_deltaEtaDivReco_MCHEPMC_EHT_" , "#Delta #eta / #eta_{reco} - EHT;#Delta #eta / #eta_{reco}"
   ,150,-1,1);
	
   //----------------phi
   // 11 delta phi
   h_deltaPhi_MCHEPMC_PF_
   = new TH1F("h_deltaPhi_MCHEPMC_PF_","#Delta #phi - PF;#Delta #phi"
   ,250,-0.2,0.2);
   h_deltaPhi_MCHEPMC_EHT_
   = new TH1F("h_deltaPhi_MCHEPMC_EHT_","#Delta #phi - EHT;#Delta #phi"
   ,250,-0.2,0.2);

   // 12 delta phi vs pt
   h_deltaPhivsPt_MCHEPMC_PF_
   = new TH1F("h_deltaPhivsPt_MCHEPMC_PF_","#Delta #phi vs p_{T} - PF;p_{T};#Delta #phi"
   ,50,0,200);
   h_deltaPhivsPt_MCHEPMC_EHT_
   = new TH1F("h_deltaPhivsPt_MCHEPMC_EHT_","#Delta #phi vs p_{T} - EHT;p_{T};#Delta #phi"
   ,50,0,200);
   n_deltaPhivsPt=new TH1F("n_deltaPhivsPt","qqq",50,0,200);

   // 13 delta phi  vs  eta
   h_deltaPhivsEta_MCHEPMC_PF_
   = new TH1F("h_deltaPhivsEta_MCHEPMC_PF_","#Delta #phi vs #eta - PF;#eta;#Delta #phi"
   ,80,-3,3);//100,-0.5,0.5
   h_deltaPhivsEta_MCHEPMC_EHT_
   = new TH1F("h_deltaPhivsEta_MCHEPMC_EHT_","#Delta #phi vs #eta - EHT;#eta;#Delta #phi"
   ,80,-3,3);
   n_deltaPhivsEta=new TH1F("n_deltaPhivsEta","qqq",80,-3,3);

   // D delta phi / phi
   h_deltaPhiDivReco_MCHEPMC_PF_
   = new TH1F("h_deltaPhiDivReco_MCHEPMC_PF_" , "#Delta #phi / #phi_{reco} - PF;#Delta #phi / phi_{reco}"
   ,200,-1,3);
   h_deltaPhiDivReco_MCHEPMC_EHT_
   = new TH1F("h_deltaPhiDivReco_MCHEPMC_EHT_" , "#Delta #phi / #phi_{reco} - EHT;#Delta #phi / #phi_{reco}"
   ,200,-1,3);

   // 14  Erec / Etrue
   h_ErecDivEtrue_MCHEPMC_PF_
   = new TH1F("h_ErecDivEtrue_MCHEPMC_PF_" , "E_{T rec} / E_{T true}  - PF;p_{T}; E_{T rec} / E_{T true}"
   ,20,0,200);
   h_ErecDivEtrue_MCHEPMC_EHT_
   = new TH1F("h_ErecDivEtrue_MCHEPMC_EHT_" , "E_{T rec} / E_{T true}  - EHT;p_{T}; E_{T rec} / E_{T true}"
   ,20,0,200);
   n_ErecDivEtrue=new TH1F("n_ErecDivEtrue_PF","qqq",20,0,200);

   h_deltaEtDivEtvsEt_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtDivEtvsEt_MCHEPMC_EHT_ -> Sumw2();
   h_deltaEtDivEtvsEta_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtDivEtvsEta_MCHEPMC_EHT_ -> Sumw2();
   h_deltaEtDivEtrecvsEt_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtDivEtrecvsEt_MCHEPMC_EHT_ -> Sumw2();
   h_deltaEtDivEtrecvsEta_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtDivEtrecvsEta_MCHEPMC_EHT_ -> Sumw2();
   h_ErecDivEtrue_MCHEPMC_PF_->Sumw2();
   h_ErecDivEtrue_MCHEPMC_EHT_->Sumw2();
   h_deltaEtavsPt_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtavsPt_MCHEPMC_EHT_ -> Sumw2();
   h_deltaEtavsEta_MCHEPMC_PF_ -> Sumw2();
   h_deltaEtavsEta_MCHEPMC_EHT_ -> Sumw2();
   h_deltaPhivsPt_MCHEPMC_PF_ -> Sumw2();
   h_deltaPhivsPt_MCHEPMC_EHT_ -> Sumw2();
   h_deltaPhivsEta_MCHEPMC_PF_ -> Sumw2();
   h_deltaPhivsEta_MCHEPMC_EHT_ -> Sumw2();
  
}

PFBenchmarkAlgo::~PFBenchmarkAlgo()
{

}

void PFBenchmarkAlgo::doBenchmark()
{
  
  cout<<"TauBenchmarkAnalyzer"
    <<"opening output root file "<<outputRootFileName_<<endl;
    
 
  // ==================================================================	
  // MAKING TRUE PARTICLES FROM HEPMC-PARTICLES =======================
  // ==================================================================
	
  double hepmc_et =0.0; 
  double hepmc_E=0.0;
  double hepmc_Z=0.0;
  double hepmc_Pt=0.0;
  double hepmc_Eta=0.0;
  double hepmc_Phi=0;

  double hepmc_pt=0.0;//Etrue
	
  int numLep = 0;
  int numNu = 0;
  bool enablegenjets_=true;
  if(enablegenjets_==true)
    {
    
      const HepMC::GenEvent *myGenEvent = hepMC_->GetEvent();
   
   
    //HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
    //cout<<&p<<endl;
    // cout<<&myGenEvent->particles_end()<<endl;
      
    //  return;
   
	  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p )
      {
		if (abs((*p)->pdg_id()) == 15) {
	  // retrieve decay vertex	    
	  hepmc_Eta=(*p)->momentum().eta();
	  hepmc_pt=sqrt((*p)->momentum().perp2());//
	  hepmc_Phi=(*p)->momentum().phi();
	  const HepMC::GenVertex * decayVtx = (*p)->end_vertex();
	  if ( decayVtx != 0 ) {
	    HepMC::GenVertex::particles_in_const_iterator child_mcpartItr  = decayVtx->particles_out_const_begin();
	     HepMC::GenVertex::particles_in_const_iterator child_mcpartItrE = decayVtx->particles_out_const_end();	    
	    for (; child_mcpartItr != child_mcpartItrE; ++child_mcpartItr) {
       	      HepMC::GenParticle * child = (*child_mcpartItr);
	      //  HepMC::Polarization * child2 = (*child_mcpartItr);
	      if (std::abs(child->pdg_id()) == 11 || std::abs(child->pdg_id()) == 13) {
		++numLep;
		return;
	      }
	      else {
	      if (std::abs(child->pdg_id())!=12 && std::abs(child->pdg_id())!=14 && std::abs(child->pdg_id())!=16 ){
	      hepmc_E=child->momentum().e();
	      hepmc_Z=child->momentum().z();
	      hepmc_Pt=sqrt(child->momentum().perp2());
	      //hepmc_Eta=child->momentum().eta();
	      ++numNu;
	      hepmc_et+=Et(hepmc_E, hepmc_Pt, hepmc_Z); //child->momentum().Et()
	      //	      LogInfo("Info")<< "pdgcode:"<<Et(hepmc_E, hepmc_Pt, hepmc_Z) 	<< endl;
	      }
	    }
	    }//loop daughter
	  }//  decayVtx
	  }//if tau//
	}
    
    //LogInfo("Info")<<"HepMC truth MC without neutrino " << hepmc_et<< endl; ;
  }//enablegenjets_
	

  

  // ==================================================================
  // GEN JETS =========================================================
  // ==================================================================
 
  double JetGEN_ETmax=0.0;
  double jetGEN_et=0.0;
  for ( unsigned int i = 0; i < genJets_->size(); i++)
    {
      //Taking the most energetic jet
      jetGEN_et=(*genJets_)[i].et();
      //if(JetGEN_ETmax<jetGEN_et)
	JetGEN_ETmax += jetGEN_et;
    }
   cout<<"biggest et of gen jets: "<<JetGEN_ETmax<<endl;
  
  // ==================================================================
  // CALO TOWER JETS (ECAL+HCAL Towers)================================
  // ==================================================================
  
  double JetEHTETmax=0.0;
  double jetcalo_et=0.0;
  double EHT_eta=0;
  double EHT_phi=0;
  for ( unsigned int i = 0; i < caloJets_->size(); ++i)
    {
      jetcalo_et=(*caloJets_)[i].et();
      //if (jetcalo_et >= JetEHTETmax) 
	JetEHTETmax += jetcalo_et;
	EHT_eta=(*caloJets_)[i].eta();
	EHT_phi=(*caloJets_)[i].phi();
    }//loop calo towers
  cout<<"biggest et of calo jets: "<<JetEHTETmax<<endl;
  
  
  // ==================================================================
  // PF Jets ==========================================================
  // ==================================================================
  
  double pfjet_et=0.0;
  double JetPFETmax=0.0;
  int jetpf_num=0;
  double PF_eta=0;
  double PF_phi=0;
  for ( unsigned int i = 0; i < (*pfJets_).size(); ++i)
    {
      pfjet_et=(*pfJets_)[i].et();
      // if (pfjet_et >= JetPFETmax) 
	JetPFETmax += pfjet_et;
	PF_eta=(*pfJets_)[i].eta();
	PF_phi=(*pfJets_)[i].phi();
    }//loop pfjets
  cout<<"biggest et of pf jets: "<<JetPFETmax<<endl;
  // ==================================================================
  // Status output ====================================================
  // ==================================================================
  
  
  cout<<"delta et calo-gen: "<<JetEHTETmax-JetGEN_ETmax<<endl;
  cout<<"delta et pflow-gen : "<<JetPFETmax-JetGEN_ETmax<<endl; 
  cout<<endl<<endl<<endl;
  
  // ==================================================================
  // Filling Histogramms ==============================================
  // ==================================================================

  h_deltaETvisible_MCHEPMC_PF_->Fill(JetPFETmax-hepmc_et);
  h_deltaETvisible_MCHEPMC_EHT_->Fill(JetEHTETmax-hepmc_et); 
	
  h_deltaEtvsEt_MCHEPMC_PF_ -> Fill(hepmc_pt,JetPFETmax-hepmc_et); 
  h_deltaEtvsEt_MCHEPMC_EHT_ -> Fill(hepmc_pt,JetEHTETmax-hepmc_et);
  n_deltaEtvsEt->Fill(hepmc_pt);
  
  h_deltaEtvsEta_MCHEPMC_PF_ -> Fill(hepmc_Eta,JetPFETmax-hepmc_et); 
  h_deltaEtvsEta_MCHEPMC_EHT_ -> Fill(hepmc_Eta,JetEHTETmax-hepmc_et);
  n_deltaEtvsEta->Fill(hepmc_Eta);

  h_deltaEtDivEtvsEt_MCHEPMC_PF_ -> Fill(hepmc_pt,(JetPFETmax-hepmc_et)/hepmc_et); 
  h_deltaEtDivEtvsEt_MCHEPMC_EHT_ -> Fill(hepmc_pt,(JetEHTETmax-hepmc_et)/hepmc_et);
  n_deltaEtDivEtvsEt->Fill(hepmc_pt);
  
  if(JetPFETmax!=0){
  h_deltaEtDivEtrecvsEt_MCHEPMC_PF_-> Fill(hepmc_pt,(JetPFETmax-hepmc_et)/JetPFETmax);}
  if(JetEHTETmax!=0){
  h_deltaEtDivEtrecvsEt_MCHEPMC_EHT_ -> Fill(hepmc_pt,(JetEHTETmax-hepmc_et)/JetEHTETmax);}//
  n_deltaEtDivEtrecvsEt->Fill(hepmc_pt);

  h_deltaEtDivEtvsEta_MCHEPMC_PF_ -> Fill(hepmc_Eta,(JetPFETmax-hepmc_et)/hepmc_et);
  h_deltaEtDivEtvsEta_MCHEPMC_EHT_ -> Fill(hepmc_Eta,(JetEHTETmax-hepmc_et)/hepmc_et);
  n_deltaEtDivEtvsEta->Fill(hepmc_Eta);

  if(JetPFETmax!=0){
  h_deltaEtDivEtrecvsEta_MCHEPMC_PF_ -> Fill(hepmc_pt,(JetPFETmax-hepmc_et)/JetPFETmax);}
  if(JetEHTETmax!=0){
  h_deltaEtDivEtrecvsEta_MCHEPMC_EHT_ -> Fill(hepmc_pt,(JetEHTETmax-hepmc_et)/JetEHTETmax);}//
  n_deltaEtDivEtrecvsEta->Fill(hepmc_Eta);

  h_deltaEta_MCHEPMC_PF_ -> Fill(PF_eta-hepmc_Eta);
  h_deltaEta_MCHEPMC_EHT_ -> Fill(EHT_eta-hepmc_Eta);

  h_deltaEtavsPt_MCHEPMC_PF_ -> Fill(hepmc_pt,PF_eta-hepmc_Eta);
  h_deltaEtavsPt_MCHEPMC_EHT_ -> Fill(hepmc_pt,EHT_eta-hepmc_Eta);
  n_deltaEtavsPt->Fill(hepmc_pt);

  h_deltaEtavsEta_MCHEPMC_PF_ -> Fill(hepmc_Eta,PF_eta-hepmc_Eta);
  h_deltaEtavsEta_MCHEPMC_EHT_ -> Fill(hepmc_Eta,EHT_eta-hepmc_Eta);
  n_deltaEtavsEta->Fill(hepmc_Eta);

  h_deltaPhi_MCHEPMC_PF_ -> Fill(PF_phi-hepmc_Phi);
  h_deltaPhi_MCHEPMC_EHT_ -> Fill(EHT_phi-hepmc_Phi);

  h_deltaPhivsPt_MCHEPMC_PF_ -> Fill(hepmc_pt,PF_phi-hepmc_Phi);
  h_deltaPhivsPt_MCHEPMC_EHT_ -> Fill(hepmc_pt,EHT_phi-hepmc_Phi);
  n_deltaPhivsPt->Fill(hepmc_pt);

  h_deltaPhivsEta_MCHEPMC_PF_ -> Fill(hepmc_Eta,PF_phi-hepmc_Phi);
  h_deltaPhivsEta_MCHEPMC_EHT_ -> Fill(hepmc_Eta,EHT_phi-hepmc_Phi);
  n_deltaPhivsEta->Fill(hepmc_Eta);

  h_deltaETDivTrue_MCHEPMC_PF_->Fill((JetPFETmax-hepmc_et)/hepmc_et);
  h_deltaETDivTrue_MCHEPMC_EHT_->Fill((JetEHTETmax-hepmc_et)/hepmc_et);

  if(JetPFETmax!=0){
  h_deltaETDivReco_MCHEPMC_PF_->Fill((JetPFETmax-hepmc_et)/JetPFETmax);}
  if(JetEHTETmax!=0){
  h_deltaETDivReco_MCHEPMC_EHT_->Fill((JetEHTETmax-hepmc_et)/JetEHTETmax);}

  h_deltaEtaDivReco_MCHEPMC_PF_ -> Fill((PF_eta-hepmc_Eta)/PF_eta);
  h_deltaEtaDivReco_MCHEPMC_EHT_ -> Fill((EHT_eta-hepmc_Eta)/EHT_eta);
	  
  h_deltaPhiDivReco_MCHEPMC_PF_ -> Fill((PF_phi-hepmc_Phi)/PF_phi);
  h_deltaPhiDivReco_MCHEPMC_EHT_ -> Fill((EHT_phi-hepmc_Phi)/EHT_phi);

  h_ErecDivEtrue_MCHEPMC_PF_ -> Fill(hepmc_pt,JetPFETmax/hepmc_et);
  h_ErecDivEtrue_MCHEPMC_EHT_ -> Fill(hepmc_pt,JetEHTETmax/hepmc_et);
  n_ErecDivEtrue->Fill(hepmc_pt);
    
}


void PFBenchmarkAlgo::createPlots()
{
  file_->Write();

}
