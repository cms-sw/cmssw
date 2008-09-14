// myJetAna.cc
// Description:  Access Cruzet Data
// Author: Frank Chlebana
// Date:  24 - July - 2008
// 
#include "RecoJets/JetAnalyzers/interface/myJetAna.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
// #include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
// #include "FWCore/Framework/interface/Handle.h"
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

#define DEBUG 1
#define MAXJETS 100


// ************************
// ************************

// Get the algorithm of the jet collections we will read from the .cfg file 
// which defines the value of the strings CaloJetAlgorithm and GenJetAlgorithm.

myJetAna::myJetAna( const ParameterSet & cfg ) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) ), 
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) )
{
}


// ************************
// ************************

void myJetAna::beginJob( const EventSetup & ) {

  edm::Service<TFileService> fs;


  // --- passed selection cuts 
  h_pt     = fs->make<TH1F>( "pt",  "Jet p_{T}", 50, 0, 1000 );
  h_et     = fs->make<TH1F>( "et",  "Jet E_{T}", 50, 0, 1000 );
  h_eta    = fs->make<TH1F>( "eta", "Jet #eta", 100, -4, 4 );
  h_phi    = fs->make<TH1F>( "phi", "Jet #phi", 50, -M_PI, M_PI );
  // ---

  dijetMass  =  fs->make<TH1F>("dijetMass","DiJet Mass",100,0,4000);

  totEneLeadJetEta1 = fs->make<TH1F>("totEneLeadJetEta1","Total Energy Lead Jet Eta1 1",100,0,1500);
  totEneLeadJetEta2 = fs->make<TH1F>("totEneLeadJetEta2","Total Energy Lead Jet Eta2 1",100,0,1500);
  totEneLeadJetEta3 = fs->make<TH1F>("totEneLeadJetEta3","Total Energy Lead Jet Eta3 1",100,0,1500);
  hadEneLeadJetEta1 = fs->make<TH1F>("hadEneLeadJetEta1","Hadronic Energy Lead Jet Eta1 1",100,0,1500);
  hadEneLeadJetEta2 = fs->make<TH1F>("hadEneLeadJetEta2","Hadronic Energy Lead Jet Eta2 1",100,0,1500);
  hadEneLeadJetEta3 = fs->make<TH1F>("hadEneLeadJetEta3","Hadronic Energy Lead Jet Eta3 1",100,0,1500);
  emEneLeadJetEta1  = fs->make<TH1F>("emEneLeadJetEta1","EM Energy Lead Jet Eta1 1",100,0,1500);
  emEneLeadJetEta2  = fs->make<TH1F>("emEneLeadJetEta2","EM Energy Lead Jet Eta2 1",100,0,1500);
  emEneLeadJetEta3  = fs->make<TH1F>("emEneLeadJetEta3","EM Energy Lead Jet Eta3 1",100,0,1500);


  hadFracEta1 = fs->make<TH1F>("hadFracEta11","Hadronic Fraction Eta1 Jet 1",100,0,1);
  hadFracEta2 = fs->make<TH1F>("hadFracEta21","Hadronic Fraction Eta2 Jet 1",100,0,1);
  hadFracEta3 = fs->make<TH1F>("hadFracEta31","Hadronic Fraction Eta3 Jet 1",100,0,1);

  SumEt  = fs->make<TH1F>("SumEt","SumEt",100,0,1000);
  MET    = fs->make<TH1F>("MET",  "MET",100,0,200);

  hf_sumTowerAllEx = fs->make<TH1F>("sumTowerAllEx","Tower Ex",100,-1000,1000);
  hf_sumTowerAllEy = fs->make<TH1F>("sumTowerAllEy","Tower Ey",100,-1000,1000);

  hf_TowerJetEt   = fs->make<TH1F>("TowerJetEt","Tower/Jet Et 1",50,0,1);

  nTowers1  = fs->make<TH1F>("nTowers1","Number of Towers pt 0.5",100,0,500);
  nTowers2  = fs->make<TH1F>("nTowers2","Number of Towers pt 1.0",100,0,500);
  nTowers3  = fs->make<TH1F>("nTowers3","Number of Towers pt 1.5",100,0,500);
  nTowers4  = fs->make<TH1F>("nTowers4","Number of Towers pt 2.0",100,0,500);

  nTowersLeadJetPt1  = fs->make<TH1F>("nTowersLeadJetPt1","Number of Towers in Lead Jet pt 0.5",100,0,200);
  nTowersLeadJetPt2  = fs->make<TH1F>("nTowersLeadJetPt2","Number of Towers in Lead Jet pt 1.0",100,0,200);
  nTowersLeadJetPt3  = fs->make<TH1F>("nTowersLeadJetPt3","Number of Towers in Lead Jet pt 1.5",100,0,200);
  nTowersLeadJetPt4  = fs->make<TH1F>("nTowersLeadJetPt4","Number of Towers in Lead Jet pt 2.0",100,0,200);

  h_nCalJets  =  fs->make<TH1F>( "nCalJets",  "Number of CalJets", 20, 0, 20 );


  h_ptCal     = fs->make<TH1F>( "ptCal",  "p_{T} of CalJet", 50, 0, 1000 );
  h_etaCal    = fs->make<TH1F>( "etaCal", "#eta of  CalJet", 100, -4, 4 );
  h_phiCal    = fs->make<TH1F>( "phiCal", "#phi of  CalJet", 50, -M_PI, M_PI );

  h_nGenJets  =  fs->make<TH1F>( "nGenJets",  "Number of GenJets", 20, 0, 20 );

  h_ptGen     =  fs->make<TH1F>( "ptGen",  "p_{T} of GenJet", 50, 0, 1000 );
  h_etaGen    =  fs->make<TH1F>( "etaGen", "#eta of GenJet", 100, -4, 4 );
  h_phiGen    =  fs->make<TH1F>( "phiGen", "#phi of GenJet", 50, -M_PI, M_PI );

  h_ptGenL    =  fs->make<TH1F>( "ptGenL",  "p_{T} of GenJetL", 50, 0, 300 );
  h_etaGenL   =  fs->make<TH1F>( "etaGenL", "#eta of GenJetL", 100, -4, 4 );
  h_phiGenL   =  fs->make<TH1F>( "phiGenL", "#phi of GenJetL", 50, -M_PI, M_PI );

  h_jetEt      = fs->make<TH1F>( "jetEt", "Total Jet Et", 100, 0, 3000 );

  h_jet1Pt    = fs->make<TH1F>( "jet1Pt", "Jet1 Pt", 100, 0, 3000 );
  h_jet2Pt    = fs->make<TH1F>( "jet2Pt", "Jet2 Pt", 100, 0, 3000 );

  h_TotalUnclusteredEt = fs->make<TH1F>( "TotalUnclusteredEt", "Total Unclustered Et", 100, 0, 500 );
  h_UnclusteredEt      = fs->make<TH1F>( "UnclusteredEt", "Unclustered Et", 100, 0, 50 );
  h_UnclusteredEts     = fs->make<TH1F>( "UnclusteredEts", "Unclustered Et", 100, 0, 2 );

  h_ClusteredE         = fs->make<TH1F>( "ClusteredE", "Clustered E", 200, 0, 20 );
  h_TotalClusteredE    = fs->make<TH1F>( "TotalClusteredE", "Total Clustered E", 200, 0, 100 );
  h_UnclusteredE       = fs->make<TH1F>( "UnclusteredE", "Unclustered E", 200, 0, 20 );
  h_TotalUnclusteredE  = fs->make<TH1F>( "TotalUnclusteredE", "Total Unclustered E", 200, 0, 100 );

  EMFraction           = fs->make<TH1F>( "EMFraction", "Jet EM Fraction", 100, 0, 1 );
  NTowers              = fs->make<TH1F>( "NTowers", "Number of Towers", 100, 0, 100 );


  h_EmEnergy   = fs->make<TH2F>( "EmEnergy",  "Em Energy",  90, -45, 45, 73, 0, 73 );
  h_HadEnergy  = fs->make<TH2F>( "HadEnergy", "Had Energy", 90, -45, 45, 73, 0, 73 );

}

// ************************
// ************************
void myJetAna::analyze( const Event& evt, const EventSetup& es ) {
 
  int EtaOk10, EtaOk13, EtaOk40;

  double LeadMass;

  float pt1;

  float minJetPt = 30.;
  float minJetPt10 = 10.;
  int jetInd, allJetInd;

  LeadMass = -1;

  math::XYZTLorentzVector p4tmp[2], p4cortmp[2];

  // --------------------------------------------------------------
  // --------------------------------------------------------------


  // **************************************************************
  // ** Loop over the two leading CaloJets and fill some histograms
  // **************************************************************
  Handle<CaloJetCollection> caloJets;
  evt.getByLabel( CaloJetAlgorithm, caloJets );

  jetInd    = 0;
  allJetInd = 0;

  EtaOk10 = 0;
  EtaOk13 = 0;
  EtaOk40 = 0;

  //  const JetCorrector* corrector = 
  //    JetCorrector::getJetCorrector (JetCorrectionService, es);

  double highestPt;
  double nextPt;

  highestPt = 0.0;
  nextPt    = 0.0;
  
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
    
    //    double scale = corrector->correction (*cal);
    double scale = 1.0;
    double corPt = scale*cal->pt();
    //    double corPt = cal->pt();
    //    cout << "Pt = " << cal->pt() << endl;
    
    if (corPt>highestPt) {
      nextPt      = highestPt;
      p4cortmp[1] = p4cortmp[0]; 
      highestPt   = corPt;
      p4cortmp[0] = scale*cal->p4();
    } else if (corPt>nextPt) {
      nextPt      = corPt;
      p4cortmp[1] = scale*cal->p4();
    }

    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt->Fill( cal->pt() );
      pt1 = cal->pt();
      p4tmp[0] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;
    }
    if (allJetInd == 2) {
      h_jet2Pt->Fill( cal->pt() );
      p4tmp[1] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;
    }

    if ( cal->pt() > minJetPt) {
      h_ptCal->Fill( cal->pt() );   
      h_etaCal->Fill( cal->eta() );
      h_phiCal->Fill( cal->phi() );
      jetInd++;
    }
  }

  h_nCalJets->Fill( jetInd ); 

  if (jetInd > 1) {
    LeadMass = (p4tmp[0]+p4tmp[1]).mass();
    dijetMass->Fill( LeadMass );    
  }


  // ******************
  // *** Jet Properties
  // ******************

  int nTow1, nTow2, nTow3, nTow4;
  //  Handle<CaloJetCollection> jets;
  //  evt.getByLabel( CaloJetAlgorithm, jets );

  // *********************************************************
  // --- Loop over jets and make a list of all the used towers
  int jjet = 0;
  for ( CaloJetCollection::const_iterator ijet=caloJets->begin(); ijet!=caloJets->end(); ijet++) {
    jjet++;

    float hadEne  = ijet->hadEnergyInHB() + ijet->hadEnergyInHO() + 
                    ijet->hadEnergyInHE() + ijet->hadEnergyInHF();                   
    float emEne   = ijet->emEnergyInEB() + ijet->emEnergyInEE() + ijet->emEnergyInHF();
    float had     = ijet->energyFractionHadronic();    

    EMFraction->Fill(ijet->emEnergyFraction());    

    float j_et = ijet->et();

    // *** Barrel
    if (fabs(ijet->eta()) < 1.3) {
      totEneLeadJetEta1->Fill(hadEne+emEne); 
      hadEneLeadJetEta1->Fill(hadEne); 
      emEneLeadJetEta1->Fill(emEne);       

      if (ijet->pt() > minJetPt10) 
	hadFracEta1->Fill(had);
    }

    // *** EndCap
    if ((fabs(ijet->eta()) > 1.3) && (fabs(ijet->eta()) < 3.) ) {

      totEneLeadJetEta2->Fill(hadEne+emEne); 
      hadEneLeadJetEta2->Fill(hadEne); 
      emEneLeadJetEta2->Fill(emEne);   
    
      if (ijet->pt() > minJetPt10) 
	hadFracEta2->Fill(had);
    }

    // *** Forward
    if (fabs(ijet->eta()) > 3.) {

      totEneLeadJetEta3->Fill(hadEne+emEne); 
      hadEneLeadJetEta3->Fill(hadEne); 
      emEneLeadJetEta3->Fill(emEne); 

      if (ijet->pt() > minJetPt10) 
	hadFracEta3->Fill(had);
    }

    // *** CaloTowers in Jet
    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();
    NTowers->Fill(nConstituents);

    if (jjet == 1) {

      nTow1 = nTow2 = nTow3 = nTow4 = 0;
      for (int i = 0; i <nConstituents ; i++){

	float et  = jetCaloRefs[i]->et();

	if (et > 0.5) nTow1++;
	if (et > 1.0) nTow2++;
	if (et > 1.5) nTow3++;
	if (et > 2.0) nTow4++;
	
	hf_TowerJetEt->Fill(et/j_et);

      }

      nTowersLeadJetPt1->Fill(nTow1);
      nTowersLeadJetPt2->Fill(nTow2);
      nTowersLeadJetPt3->Fill(nTow3);
      nTowersLeadJetPt4->Fill(nTow4);

    }

  }


  // **********************
  // *** Unclustered Energy
  // **********************

  double SumPtJet(0);

  double SumEtNotJets(0);
  double SumEtJets(0);
  double SumEtTowers(0);
  double TotalClusteredE(0);
  double TotalUnclusteredE(0);

  double sumJetPx(0);
  double sumJetPy(0);

  double sumTowerAllPx(0);
  double sumTowerAllPy(0);

  double sumTowerAllEx(0);
  double sumTowerAllEy(0);

  std::vector<CaloTowerPtr>   UsedTowerList;
  std::vector<CaloTower>      TowerUsedInJets;
  std::vector<CaloTower>      TowerNotUsedInJets;


  // *********************
  // *** CaloTowers
  // *********************
  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( "towerMaker", caloTowers );

  nTow1 = nTow2 = nTow3 = nTow4 = 0;

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  //  double sum_ez = 0.0;

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    Double_t  et = tower->et();

    if (et > 0.5) nTow1++;
    if (et > 1.0) nTow2++;
    if (et > 1.5) nTow3++;
    if (et > 2.0) nTow4++;

    //    if ( (fabs(tower->ieta() > 42)) ||  (fabs(tower->iphi()) > 72) ) {
    //      std::cout << "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() << std::endl;
    //    }

    if (tower->emEnergy() > 2.0) {
      h_EmEnergy->Fill (tower->ieta(), tower->iphi(), tower->emEnergy());
    }
    if (tower->hadEnergy() > 2.0) {
      h_HadEnergy->Fill (tower->ieta(), tower->iphi(), tower->hadEnergy());
    }


    if (et>0.5) {

      // ********
      double phix   = tower->phi();
      //      double theta = tower->theta();
      //      double e     = tower->energy();
      //      double et    = e*sin(theta);
      //      double et    = tower->emEt() + tower->hadEt();
      double et    = tower->et();

      //      sum_ez += e*cos(theta);
      sum_et += et;
      sum_ex += et*cos(phix);
      sum_ey += et*sin(phix);
      // ********

      Double_t phi = tower->phi();
      SumEtTowers += tower->et();

      sumTowerAllEx += et*cos(phi);
      sumTowerAllEy += et*sin(phi);

    }

  }

  SumEt->Fill(sum_et);
  MET->Fill(sqrt( sum_ex*sum_ex + sum_ey*sum_ey));

  hf_sumTowerAllEx->Fill(sumTowerAllEx);
  hf_sumTowerAllEy->Fill(sumTowerAllEy);

  nTowers1->Fill(nTow1);
  nTowers2->Fill(nTow2);
  nTowers3->Fill(nTow3);
  nTowers4->Fill(nTow4);


  // *********************
  // *********************

  UsedTowerList.clear();
  TowerUsedInJets.clear();
  TowerNotUsedInJets.clear();

  // --- Loop over jets and make a list of all the used towers
  //  evt.getByLabel( CaloJetAlgorithm, jets );
  for ( CaloJetCollection::const_iterator ijet=caloJets->begin(); ijet!=caloJets->end(); ijet++) {

    Double_t jetPt  = ijet->pt();
    Double_t jetPhi = ijet->phi();

    //    if (jetPt>5.0) {

      Double_t jetPx = jetPt*cos(jetPhi);
      Double_t jetPy = jetPt*sin(jetPhi);

      sumJetPx +=jetPx;
      sumJetPy +=jetPy;

      const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
      int nConstituents = jetCaloRefs.size();
      for (int i = 0; i <nConstituents ; i++){
        UsedTowerList.push_back(jetCaloRefs[i]);
      }

      SumPtJet +=jetPt;

    //    }

  }

  int NTowersUsed = UsedTowerList.size();

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    CaloTower  t = *tower;
    Double_t  et = tower->et();

    if(et>0) {

      Double_t phi = tower->phi();
      SumEtTowers += tower->et();

      sumTowerAllPx += et*cos(phi);
      sumTowerAllPy += et*sin(phi);

      bool used = false;

      for(int i=0; i<NTowersUsed; i++){
        if(tower->id() == UsedTowerList[i]->id()){
          used=true;
          break;
        }
      }

      if (used) {
        TowerUsedInJets.push_back(t);
      } else {
        TowerNotUsedInJets.push_back(t);
      }
    }
  }

  int nUsed    = TowerUsedInJets.size();
  int nNotUsed = TowerNotUsedInJets.size();

  SumEtJets    = 0;
  SumEtNotJets = 0;
  TotalClusteredE   = 0;
  TotalUnclusteredE = 0;

  for(int i=0;i<nUsed;i++){
    SumEtJets += TowerUsedInJets[i].et();
    h_ClusteredE->Fill(TowerUsedInJets[i].energy());
    if (TowerUsedInJets[i].energy() > 1.0) 
      TotalClusteredE += TowerUsedInJets[i].energy();
  }
  h_jetEt->Fill(SumEtJets);

  for(int i=0;i<nNotUsed;i++){
    if (TowerNotUsedInJets[i].et() > 0.5)
      SumEtNotJets += TowerNotUsedInJets[i].et();
    h_UnclusteredEt->Fill(TowerNotUsedInJets[i].et());
    h_UnclusteredEts->Fill(TowerNotUsedInJets[i].et());
    h_UnclusteredE->Fill(TowerNotUsedInJets[i].energy());
    if (TowerNotUsedInJets[i].energy() > 1.0)  
      TotalUnclusteredE += TowerNotUsedInJets[i].energy();
  }

  h_TotalClusteredE->Fill(TotalClusteredE);
  h_TotalUnclusteredE->Fill(TotalUnclusteredE);
  h_TotalUnclusteredEt->Fill(SumEtNotJets);


  // ********************************
  // *** Events passing seletion cuts
  // ********************************

  // --- Cosmic Cleanup
  // --- Vertex
  // --- Eta 

  for( CaloJetCollection::const_iterator ijet = caloJets->begin(); ijet != caloJets->end(); ++ ijet ) {
    
    if ( (fabs(ijet->eta()) < 1.3) && 
	 (fabs(ijet->pt())  > 20.) &&
	 (ijet->emEnergyFraction() > 0.01) &&
	 (ijet->emEnergyFraction() > 0.99) ) {
      h_pt->Fill(ijet->pt());
      h_et->Fill(ijet->et());
      h_eta->Fill(ijet->eta());
      h_phi->Fill(ijet->phi());
      
    }    
  }



  //*****************************
  //*** Get the GenJet collection
  //*****************************

      /**************
  Handle<GenJetCollection> genJets;
  evt.getByLabel( GenJetAlgorithm, genJets );

  //Loop over the two leading GenJets and fill some histograms
  jetInd    = 0;
  allJetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
    allJetInd++;
    if (allJetInd == 1) {
      p4tmp[0] = gen->p4();
    }
    if (allJetInd == 2) {
      p4tmp[1] = gen->p4();
    }

    if ( (allJetInd == 1) || (allJetInd == 2) ) {
      h_ptGenL->Fill( gen->pt() );
      h_etaGenL->Fill( gen->eta() );
      h_phiGenL->Fill( gen->phi() );
    }

    if ( gen->pt() > minJetPt) {
      // std::cout << "GEN JET1 #" << jetInd << std::endl << gen->print() << std::endl;
      h_ptGen->Fill( gen->pt() );
      h_etaGen->Fill( gen->eta() );
      h_phiGen->Fill( gen->phi() );
      jetInd++;
    }
  }

  h_nGenJets->Fill( jetInd );
      *******/


}

// ***********************************
// ***********************************
void myJetAna::endJob() {

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myJetAna);
