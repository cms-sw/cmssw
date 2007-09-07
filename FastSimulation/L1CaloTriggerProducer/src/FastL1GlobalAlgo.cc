// -*- C++ -*-
//
// Package:    L1CaloTriggerProducer
// Class:      FastL1GlobalAlgo
// 
/**\class FastL1GlobalAlgo

 Description: Global algorithm.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Feb 19 13:25:24 CST 2007
// $Id: FastL1GlobalAlgo.cc,v 1.17 2007/08/23 04:48:42 chinhan Exp $
//

// No BitInfos for release versions

#include "FastSimulation/L1CaloTriggerProducer/interface/FastL1GlobalAlgo.h"
#include <iostream>
#include <fstream>

//
// constructors and destructor
//
FastL1GlobalAlgo::FastL1GlobalAlgo(const edm::ParameterSet& iConfig)
{  
  m_RMap = FastL1RegionMap::getL1RegionMap();

  // Get L1 config
  m_L1Config.DoEMCorr = iConfig.getParameter<bool>("DoEMCorr");
  m_L1Config.DoJetCorr = iConfig.getParameter<bool>("DoJetCorr");

  // get uncompressed hcal et
  m_L1Config.HcalLUT = iConfig.getParameter<edm::FileInPath>("HcalLUT");

  m_L1Config.EMSeedEnThreshold = iConfig.getParameter<double>("EMSeedEnThreshold");
  m_L1Config.EMActiveLevel = iConfig.getParameter<double>("EMActiveLevel");
  m_L1Config.HadActiveLevel = iConfig.getParameter<double>("HadActiveLevel");
  m_L1Config.noTauVetoLevel = iConfig.getParameter<double>("noTauVetoLevel");	
  m_L1Config.hOeThreshold = iConfig.getParameter<double>("hOeThreshold");
  m_L1Config.FGEBThreshold = iConfig.getParameter<double>("FGEBThreshold");
  m_L1Config.FGEEThreshold = iConfig.getParameter<double>("FGEEThreshold");

  m_L1Config.MuonNoiseLevel = iConfig.getParameter<double>("MuonNoiseLevel");
  m_L1Config.EMNoiseLevel = iConfig.getParameter<double>("EMNoiseLevel");
  m_L1Config.HadNoiseLevel = iConfig.getParameter<double>("HadNoiseLevel");
  m_L1Config.QuietRegionThreshold = iConfig.getParameter<double>("QuietRegionThreshold");
  m_L1Config.JetSeedEtThreshold = iConfig.getParameter<double>("JetSeedEtThreshold");

  m_L1Config.TowerEMLSB = iConfig.getParameter<double>("TowerEMLSB");
  m_L1Config.TowerHadLSB = iConfig.getParameter<double>("TowerHadLSB");
  m_L1Config.EMLSB = iConfig.getParameter<double>("EMLSB");
  m_L1Config.JetLSB = iConfig.getParameter<double>("JetLSB");

  m_L1Config.CrystalEBThreshold = iConfig.getParameter<double>("CrystalEBThreshold");
  m_L1Config.CrystalEEThreshold = iConfig.getParameter<double>("CrystalEEThreshold");

  m_L1Config.TowerEBThreshold = iConfig.getParameter<double>("TowerEBThreshold");
  m_L1Config.TowerEEThreshold = iConfig.getParameter<double>("TowerEEThreshold");
  m_L1Config.TowerHBThreshold = iConfig.getParameter<double>("TowerHBThreshold");
  m_L1Config.TowerHEThreshold = iConfig.getParameter<double>("TowerHEThreshold");

  m_L1Config.TowerEBScale = iConfig.getParameter<double>("TowerEBScale");
  m_L1Config.TowerEEScale = iConfig.getParameter<double>("TowerEEScale");
  m_L1Config.TowerHBScale = iConfig.getParameter<double>("TowerHBScale");
  m_L1Config.TowerHEScale = iConfig.getParameter<double>("TowerHEScale");

  m_L1Config.noFGThreshold = iConfig.getParameter<double>("noFGThreshold");	
  
  m_L1Config.EmInputs = iConfig.getParameter <std::vector<edm::InputTag> >("EmInputs");
  m_L1Config.TowerInput = iConfig.getParameter<edm::InputTag>("TowerInput");

  m_L1Config.EcalTPInput = iConfig.getParameter<edm::InputTag>("EcalTPInput");
  m_L1Config.HcalTPInput = iConfig.getParameter<edm::InputTag>("HcalTPInput");


  //Load Hcal LUT file
  std::ifstream userfile;
  const std::string userfileName = m_L1Config.HcalLUT.fullPath();
  userfile.open(userfileName.c_str());
  static const int etabound = 32;
  static const int tpgmax = 256;
  for (int i=0; i<tpgmax; i++) { 
    for(int j = 1; j <=etabound; j++) {
      userfile >> m_hcaluncomp[j][i];
      //std::cout<<"+++ "<<m_hcaluncomp[j][i]<<std::endl;
    }
  }
  userfile.close();  
  

}

FastL1GlobalAlgo::~FastL1GlobalAlgo()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//


// ------------ Dump out CaloTower info  ------------
void
FastL1GlobalAlgo::CaloTowersDump(edm::Event const& e) {
  /*
    std::vector<edm::Handle<CaloTowerCollection> > prods;
    
    edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << "Start!";
    e.getManyByType(prods);
    
    std::vector<edm::Handle<CaloTowerCollection> >::iterator i;

    for (i=prods.begin(); i!=prods.end(); i++) {
      const CaloTowerCollection& c=*(*i);
      
      for (CaloTowerCollection::const_iterator j=c.begin(); j!=c.end(); j++) {
	edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << *j;
	}
      }
   edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << "End!";
  */


  edm::Handle<CaloTowerCollection> input;
    
  edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << "Start!";
  e.getByLabel(m_L1Config.TowerInput,input);

  for (CaloTowerCollection::const_iterator j=input->begin(); j!=input->end(); j++) {
    edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << *j;
  }

  edm::LogInfo("FastL1GlobalAlgo::CaloTowersDump") << "End!";
}


// ------------ For sort()  ------------
namespace myspace {
  bool 
  //FastL1GlobalAlgo::greaterEt( const reco::Candidate& a, const reco::Candidate& b ) {
  greaterEt( const reco::Candidate& a, const reco::Candidate& b ) {
    return (a.et()>b.et());
  }
}

// ------------ Find Jets  ------------
void
FastL1GlobalAlgo::findJets() {

  m_TauJets.clear();
  m_CenJets.clear();
  m_ForJets.clear();

  // only barrel right now!
  // barrel: 7-14 with 22 ieta
  //for (int i=0; i<396 ; i++) {
  for (int i=0; i<396; i++) {
    // barrel/endcap part only right now
    //if ((i%22)>3 && (i%22)<18) {
    // Test HF!!!
  std::pair<double, double> p = m_RMap->getRegionCenterEtaPhi(i);

  double eta   = p.first;
  double phi   = p.second;

  m_Regions[i].BitInfo.setEta ( eta);
  m_Regions[i].BitInfo.setPhi ( phi);

    if (m_Regions.at(i).SumEt()>m_L1Config.JetSeedEtThreshold) {
      if (isMaxEtRgn_Window33(i)) {
	if (isTauJet(i)) {
	  addJet(i,true);    
	} else {
m_Regions[i].BitInfo.setIsolationVeto ( true);
	  addJet(i,false);    
	}
      }
else {m_Regions[i].BitInfo.setMaxEt ( true);}
    } else {
      m_Regions[i].BitInfo.setSumEtBelowThres ( true);
    }
    //}
  }
  
}


// ------------ Add a jet  ------------
void
FastL1GlobalAlgo::addJet(int iRgn, bool taubit) {
  std::pair<double, double> p = m_RMap->getRegionCenterEtaPhi(iRgn);

  //double e     = m_Regions.at(iRgn).GetJetE();
  //double et = GCTEnergyTrunc(m_Regions.at(iRgn).GetJetEt(), m_L1Config.JetLSB, false);
  double et = m_Regions.at(iRgn).GetJetEt();

  double eta   = p.first;
  double phi   = p.second;

  /*
  if (et>0.) {
    std::cout<<">>> "<<et<<" "<<eta<<" "<<phi<<std::endl;
  }
  */

  if (m_L1Config.DoJetCorr) {
    et = GCTEnergyTrunc(corrJetEt(et,eta), m_L1Config.JetLSB, false);
  } else {
    et = GCTEnergyTrunc(et, m_L1Config.JetLSB, false);
  }
  /*
  if (et>0.) {
    std::cout<<"*** "<<et<<" "<<eta<<" "<<phi<<std::endl;
  }
  */

  double theta = 2.*atan(exp(-eta));
  double ex = et*cos(phi);
  double ey = et*sin(phi);
  //double ex = e*sin(theta)*cos(phi);
  //double ey = e*sin(theta)*sin(phi);
  double e = ex/sin(theta)/cos(phi);
  double ez = e*cos(theta);

//sm
m_Regions[iRgn].BitInfo.setEt ( et);
m_Regions[iRgn].BitInfo.setEnergy ( e);
//ms


  reco::Particle::LorentzVector rp4(ex,ey,ez,e); 
  l1extra::L1JetParticle tjet(rp4);
  
  //if (et>0.) {
  //  std::cout << "region et, eta, phi: "<< et << " "<< eta << " "<< phi << " " << std::endl;
  //  std::cout << "reg lv et, eta, phi: "<< tjet.et()<<" "<< tjet.eta()<<" "<< tjet.phi()<<" " << std::endl;
  //}

  if (et>=2.) { // set from 5. by hands
    if ((taubit || et>m_L1Config.noTauVetoLevel) && (std::abs(eta)<3.0) ) {
      m_TauJets.push_back(tjet);
      // sort by et 
      std::sort(m_TauJets.begin(),m_TauJets.end(), myspace::greaterEt);
    } else {
//sm
m_Regions[iRgn].BitInfo.setSoft ( true);
//ms
      if (std::abs(eta)<3.0) {
	m_CenJets.push_back(tjet);
	std::sort(m_CenJets.begin(),m_CenJets.end(), myspace::greaterEt);
      } else {
	m_ForJets.push_back(tjet);
	std::sort(m_ForJets.begin(),m_ForJets.end(), myspace::greaterEt);
      }
    }
  }
//else{m_Regions[iRgn].BitInfo.setSoft ( true);} 
}


// ------------ Fill Egammas for TP input------------
void
FastL1GlobalAlgo::FillEgammasTP(edm::Event const& e) {
  m_Egammas.clear();
  m_isoEgammas.clear();

  for (int i=0; i<396; i++) { 
    CaloTowerCollection towers = m_Regions[i].GetCaloTowers();

    for (CaloTowerCollection::const_iterator cnd=towers.begin(); cnd!=towers.end(); cnd++) {
      if (cnd->emEt()<0.1 && cnd->hadEt()<0.1) continue;

      reco::Particle::LorentzVector rp4(0.,0.,0.,0.);
      l1extra::L1EmParticle* ph = new l1extra::L1EmParticle(rp4);
      CaloTowerDetId cid   = cnd->id();
      //std::cout<<"+++ "<<cid<<std::endl;
      
      int emTag = isEMCand(cid,ph,e);
      //std::cout<<"+++ "<<ph<<std::endl;
      
      // 1 = non-iso EM, 2 = iso EM
      if (emTag==1) {
	m_Egammas.push_back(*ph);
      } else if (emTag==2) {
	m_isoEgammas.push_back(*ph);
      }
      
    }
    std::sort(m_Egammas.begin(),m_Egammas.end(), myspace::greaterEt);
    std::sort(m_isoEgammas.begin(),m_isoEgammas.end(), myspace::greaterEt);
  }
}


// ------------ Fill Egammas for Tower input ------------
void
FastL1GlobalAlgo::FillEgammas(edm::Event const& e) {
  m_Egammas.clear();
  m_isoEgammas.clear();

  //std::vector< edm::Handle<CaloTowerCollection> > input;
  //e.getManyByType(input);
  edm::Handle<CaloTowerCollection> input;
  e.getByLabel(m_L1Config.TowerInput,input);

  //std::vector< edm::Handle<CaloTowerCollection> >::iterator j;
  //for (j=input.begin(); j!=input.end(); j++) {
  //  const CaloTowerCollection& c=*(*j);
    
  //for (CaloTowerCollection::const_iterator cnd=c.begin(); cnd!=c.end(); cnd++) {
    for (CaloTowerCollection::const_iterator cnd=input->begin(); cnd!=input->end(); cnd++) {
      reco::Particle::LorentzVector rp4(0.,0.,0.,0.);
      l1extra::L1EmParticle* ph = new l1extra::L1EmParticle(rp4);
      CaloTowerDetId cid   = cnd->id();

      int emTag = isEMCand(cid,ph,e);
      
      // 1 = non-iso EM, 2 = iso EM
      if (emTag==1) {
	m_Egammas.push_back(*ph);
      } else if (emTag==2) {
	m_isoEgammas.push_back(*ph);
      }

    }
  //}

  std::sort(m_Egammas.begin(),m_Egammas.end(), myspace::greaterEt);
  std::sort(m_isoEgammas.begin(),m_isoEgammas.end(), myspace::greaterEt);
}

// ------------ Fill MET 1: loop over towers ------------
void
FastL1GlobalAlgo::FillMET(edm::Event const& e) {
  //std::vector< edm::Handle<CaloTowerCollection> > input;
  //e.getManyByType(input);
  edm::Handle<CaloTowerCollection> input;
  e.getByLabel(m_L1Config.TowerInput,input);

  double sum_hade = 0.0;
  double sum_hadet = 0.0;
  double sum_hadex = 0.0;
  double sum_hadey = 0.0;
  double sum_hadez = 0.0;
  double sum_e = 0.0;
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;

  //std::vector< edm::Handle<CaloTowerCollection> >::iterator i;

  //for (i=input.begin(); i!=input.end(); i++) {
  //  const CaloTowerCollection& c=*(*i);

  //  for (CaloTowerCollection::const_iterator candidate=c.begin(); candidate!=c.end(); candidate++) {
  for (CaloTowerCollection::const_iterator candidate=input->begin(); candidate!=input->end(); candidate++) {
      //double eme    = candidate->emEnergy();
      //double hade    = candidate->hadEnergy();
      double eme    = candidate->emEt();
      double hade    = candidate->hadEt();
      
      double EThres = 0.;
      double HThres = 0.;
      double EBthres = m_L1Config.TowerEBThreshold;
      double HBthres = m_L1Config.TowerHBThreshold;
      double EEthres = m_L1Config.TowerEBThreshold;
      double HEthres = m_L1Config.TowerEEThreshold;

      //if(std::abs(candidate->eta())<1.479) {
      if(std::abs(candidate->eta())<2.322) {
	EThres = EBthres;
      } else {
	EThres = EEthres;
      }
      //if(std::abs(candidate->eta())<1.305) {
      if(std::abs(candidate->eta())<2.322) {
	HThres = HBthres;
      } else {
	HThres = HEthres;
      }

      // rescale energies
      double emScale = 1.0;
      double hadScale = 1.0;
      if (std::abs(candidate->eta()>1.3050) && std::abs(candidate->eta())<3.0) {
	hadScale = m_L1Config.TowerHEScale;
	emScale = m_L1Config.TowerEEScale;
      }
      if (std::abs(candidate->eta()<1.3050)) {
	hadScale = m_L1Config.TowerHBScale;
	emScale = m_L1Config.TowerEBScale;
      }
      eme    *= emScale;
      hade   *= hadScale;
      
      if (eme>=EThres || hade>=HThres) {
	double phi   = candidate->phi();
	double eta = candidate->eta();
	//double et    = candidate->et();
	//double e     = candidate->energy();
	double theta = 2.*atan(exp(-eta));
	double et    = 0.;
	double e     = 0.;
	double had_et    = 0.;
	double had_e     = 0.;

	if (eme>=EThres) {
	  et    += candidate->emEt();
	  e    += candidate->emEnergy();
	}
	if (hade>=HThres) {
 	  et    += candidate->hadEt();
	  e    += candidate->hadEnergy();
 	  had_et  += candidate->hadEt();
	  had_e   += candidate->hadEnergy();
	}

	//sum_et += et;
	sum_et += RCTEnergyTrunc(et,1.0,1024);
	sum_ex += et*cos(phi);
	sum_ey += et*sin(phi); 
	//sum_ex += e*sin(theta)*cos(phi);
	//sum_ey += e*sin(theta)*sin(phi); 
	//sum_e += e;
	sum_e += et/sin(theta);
	sum_ez += et*cos(theta)/sin(theta);

	sum_hadet += had_et;
	sum_hadex += had_et*cos(phi);
	sum_hadey += had_et*sin(phi); 
	//sum_hadex += had_e*sin(theta)*cos(phi);
	//sum_hadey += had_e*sin(theta)*sin(phi); 
	//sum_hade += had_e;
	sum_hade += had_et/sin(theta);
	sum_hadez += had_et*cos(theta)/sin(theta);
      }
    }
  //}

  reco::Particle::LorentzVector rp4(-sum_ex,-sum_ey,0.,std::sqrt(sum_ex*sum_ex + sum_ey*sum_ey));
  m_MET = l1extra::L1EtMissParticle(rp4,sum_et,0.);  
}

// ------------ Fill MET 2: loop over regions ------------
void
FastL1GlobalAlgo::FillMET() {
  double sum_e = 0.0;
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;

  for (int i=0; i<396; i++) { 
    std::pair<double, double> etaphi = m_RMap->getRegionCenterEtaPhi(i);
    double phi   = etaphi.second;
    double eta = etaphi.first;
    double theta = 2.*atan(exp(-eta));
    
    double et = m_Regions[i].SumEt();
    //double e = m_Regions[i].SumE();

    //sum_et += et;
    sum_et += RCTEnergyTrunc(et,1.0,1024);
    sum_ex += et*cos(phi);
    sum_ey += et*sin(phi); 
    //sum_ex += e*sin(theta)*cos(phi);
    //sum_ey += e*sin(theta)*sin(phi); 
    //sum_e += e;
    sum_e += et/sin(theta);
    sum_ez += et*cos(theta)/sin(theta);
    //sum_ez += e*cos(theta);
    //sum_et += e*sin(theta);
  }
  
  //reco::Particle::LorentzVector rp4(-sum_ex,-sum_ey,-sum_ez,sum_e);
  reco::Particle::LorentzVector rp4(-sum_ex,-sum_ey,0.,std::sqrt(sum_ex*sum_ex + sum_ey*sum_ey));
  //edm::LogInfo("********** FastL1GlobalAlgo::FillMET()")<<rp4.mass()<<std::endl; 
  m_MET = l1extra::L1EtMissParticle(rp4,sum_et,0.);  
  
}

void 
FastL1GlobalAlgo::InitL1Regions() 
{
  m_Regions.clear();
  m_Regions = std::vector<FastL1Region>(396); // ieta: 1-22, iphi: 1-18

  // init regions
  for (int i=0; i<396; i++) {
    m_Regions[i].SetParameters(m_L1Config);
/*/ sm
for (unsigned int j = 0; j < 16; ++j){
 m_Regions[i].BitInfo.ecal[j];
 m_Regions[i].BitInfo.hcal[j];
}
//ms */     
    std::pair<int, int> p = m_RMap->getRegionEtaPhiIndex(i);
    m_Regions[i].SetEtaPhiIndex(p.first,p.second,i);
    CaloTower c;
    //std::cout<<"+++ "<<c.emEt()<<" "<<c.hadEt()<<std::endl;
    for (int twrid=0; twrid<16; twrid++) {
      m_Regions[i].FillTowerZero(c,twrid);
    }
  }  
}

// ------------ Fill L1 Regions on Trigger Primitives------------
void 
FastL1GlobalAlgo::FillL1RegionsTP(edm::Event const& e, const edm::EventSetup& s) 
{
  //edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  //s.get<IdealGeometryRecord>().get(cttopo);
  //const CaloTowerConstituentsMap* theTowerConstituentsMap = cttopo.product();

  InitL1Regions();

  edm::Handle<EcalTrigPrimDigiCollection> ETPinput;
  e.getByLabel(m_L1Config.EcalTPInput,ETPinput);

  edm::Handle<HcalTrigPrimDigiCollection> HTPinput;
  e.getByLabel(m_L1Config.HcalTPInput,HTPinput);

  //CaloTowerCollection towers;
  int hEtV[396][16] = {0};
  int hFGV[396][16] = {0};
  int hiEtaV[396][16] = {0};
  int hiPhiV[396][16] = {0};
  for (HcalTrigPrimDigiCollection::const_iterator hTP=HTPinput->begin(); 
       hTP!=HTPinput->end(); hTP++) {
    
    //if (hTP!=HTPinput->end()) {
    HcalTrigTowerDetId htwrid   = hTP->id();
    //DetId htwrid   = (DetId)hTP->id();
    //std::cout<<"******* AAAAAAAAAAA"<<std::endl;
    //CaloTowerDetId hTPtowerDetId = theTowerConstituentsMap->towerOf((DetId)htwrid);
    //std::cout<<"******* BBBBBBBBB"<<std::endl;
    
    int rgnid = 999;
    int twrid = 999;
    rgnid  = m_RMap->getRegionIndex(htwrid.ieta(),htwrid.iphi());
    twrid = m_RMap->getRegionTowerIndex(htwrid.ieta(),htwrid.iphi());

    /*
    if (hTP->SOI_compressedEt()>0) {
      std::cout<<">>> "<<hTP->SOI_compressedEt()<<" "<<hTP->SOI_fineGrain()<<" "
    	       <<rgnid<<" "<<twrid<<std::endl;
    }
    */

    hEtV[rgnid][twrid] = (int)hTP->SOI_compressedEt();
    hFGV[rgnid][twrid] = (int)hTP->SOI_fineGrain();
    hiEtaV[rgnid][twrid] = (int)htwrid.ieta();
    hiPhiV[rgnid][twrid] = (int)htwrid.iphi();
    
  }
  

  int emEtV[396][16] = {0};
  int emFGV[396][16] = {0};
  int emiEtaV[396][16] = {0};
  int emiPhiV[396][16] = {0};
  double emEtaV[396][16] = {0.};
  for (EcalTrigPrimDigiCollection::const_iterator eTP=ETPinput->begin(); 
       eTP!=ETPinput->end(); eTP++) {
    int rgnid = 999;
    int twrid = 999;
   
    int hiphi = eTP->id().iphi();
    rgnid  = m_RMap->getRegionIndex(eTP->id().ieta(),hiphi);
    twrid = m_RMap->getRegionTowerIndex(eTP->id().ieta(),hiphi);
  
    /*  
    if (eTP->compressedEt()>0) {
      std::cout<<"*** "<<eTP->compressedEt()<<" "<<eTP->fineGrain()<<" "
    	       <<rgnid<<" "<<twrid<<std::endl;
    }
    */

    emEtV[rgnid][twrid] = (int)eTP->compressedEt();
    emFGV[rgnid][twrid] = (int)eTP->fineGrain();
    emiEtaV[rgnid][twrid] = (int)eTP->id().ieta();
    emiPhiV[rgnid][twrid] = (int)eTP->id().iphi();


    edm::ESHandle<CaloGeometry> cGeom; 
    s.get<IdealGeometryRecord>().get(cGeom);    
    const GlobalPoint gP1 = cGeom->getPosition(eTP->id());
    double eta = gP1.eta();  
    //double phi = gP1.phi();    
    emEtaV[rgnid][twrid] = eta;
  }



  for (int i=0; i<396; i++) {
    for (int j=0; j<16; j++) {
/* sm
if  (emEtV[i][j] == 0 && hEtV[i][j] == 0){
m_Regions[i].BitInfo.ecal.push_back(0);
m_Regions[i].BitInfo.hcal.push_back(0);
}
*///ms
      if (emEtV[i][j]>0 || hEtV[i][j]>0) {
	

	std::pair<double, double> etaphi 
	  = m_RMap->getRegionCenterEtaPhi(i);
	double eta = etaphi.first;
	double phi = etaphi.second;
   

 	double emEt = ((double) emEtV[i][j]) * m_L1Config.EMLSB;
	//double hadEt = ((double )hEtV[i][j]) * m_L1Config.JetLSB * cos(2.*atan(exp(-hiEtaV[i][j])));
	int iAbsTwrEta = std::abs(hiEtaV[i][j]);
	double hadEt = ((double )hcaletValue(iAbsTwrEta, hEtV[i][j])) * m_L1Config.JetLSB;
   
	if (m_L1Config.DoEMCorr) {
	  emEt = corrEmEt(emEt,emEtaV[i][j]);
	}

	double et = emEt + hadEt;
/*/ sm
m_Regions[i].BitInfo.ecal[j] = emEt;
m_Regions[i].BitInfo.hcal[j] = hadEt;
//ms */
	math::RhoEtaPhiVector lvec(et,eta,phi);
	
	CaloTowerDetId towerDetId;  
	if (emEtV[i][j]>0) 
 	  towerDetId = CaloTowerDetId(emiEtaV[i][j],emiPhiV[i][j]); 
	else
	  towerDetId = CaloTowerDetId(hiEtaV[i][j],hiPhiV[i][j]); 
	  
	CaloTower t = CaloTower(towerDetId, lvec, 
				emEt, hadEt, 
				0., 0, 0);
    
	m_Regions[i].FillTower_Scaled(t,j,false);
    
	/*
	if (et>0) {
	  std::cout<<"+++ "<<emEt<<" "<<hadEt<<" "
		   <<i<<" "<<j<<" "
		   <<std::endl<<"-- "
		   <<t.emEt()<<" "<<t.hadEt()<<" "
		   <<t.eta()<<" "<<t.phi()<<" "
		   <<std::endl;
	}
	*/


	// Explicitely take care of integer boolean conversion
	if (emEt > 3.0 && emEt < m_L1Config.noFGThreshold && (int)emFGV[i][j]!=0) 
	  m_Regions[i].SetFGBit(j,true);
	else 
	  m_Regions[i].SetFGBit(j,false);
	
	if (emEt > 3.0 && emEt < 60. && hadEt/emEt >  m_L1Config.hOeThreshold)  
	  m_Regions[i].SetHOEBit(j,true);
	else
	  m_Regions[i].SetHOEBit(j,false);
	
	m_Regions[i].SetRegionBits(e);
      }
      

    }
  }




  

}

// ------------ Fill L1 Regions on Towers and RecHits------------
void 
FastL1GlobalAlgo::FillL1Regions(edm::Event const& e, const edm::EventSetup& iConfig) 
{
  InitL1Regions();

  // ascii visualisation of mapping
  //m_RMap->display();
  
 
  // works for barrel/endcap region only right now!
  //std::vector< edm::Handle<CaloTowerCollection> > input;
  //e.getManyByType(input);
  edm::Handle<CaloTowerCollection> input;
  e.getByLabel(m_L1Config.TowerInput,input);

  //std::vector< edm::Handle<CaloTowerCollection> >::iterator j;
  //for (j=input.begin(); j!=input.end(); j++) {
  //  const CaloTowerCollection& c=*(*j);
    
  //  for (CaloTowerCollection::const_iterator cnd=c.begin(); cnd!=c.end(); cnd++) {
  for (CaloTowerCollection::const_iterator cnd=input->begin(); cnd!=input->end(); cnd++) {

      CaloTowerDetId cid   = cnd->id();
      std::pair<int, int> pep = m_RMap->getRegionEtaPhiIndex(cid);

      //if (abs(cid.ieta())>=0) {
      //std::cout << cid.ieta() << " " << cid.iphi() << std::endl;
      //std::cout << cnd->eta() << " " << cnd->phi() << std::endl;
      //std::cout << pep.first << " " << pep.second << std::endl;
      //std::cout << "**************************************************" << std::endl;
      //}
 
      int rgnid = 999;
      int twrid = 999;
      
      //std::cout << "FastL1GlobalAlgo::FillL1Regions calotowers dump: " << *cnd << std::endl;
      
      if (std::abs(pep.first)<=22) {
	rgnid = pep.second*22 + pep.first;
	twrid = m_RMap->getRegionTowerIndex(cid);
	//std::cout << rgnid << " " << twrid << std::endl;
      } 

      if (rgnid<396 && twrid<16) {
	m_Regions[rgnid].FillTower_Scaled(*cnd,twrid);
	m_Regions[rgnid].SetRegionBits(e);
      } else {
	//std::cerr << "FastL1GlobalAlgo::FillL1Regions(): ERROR - invalid region or tower ID: " << rgnid << " | " << twrid  << std::endl;
      }

  }

  //}
  
  // Fill EM Crystals
  for (int i=0; i<396; i++) {
    m_Regions[i].FillEMCrystals(e,iConfig,m_RMap);
  }

  //checkMapping();
  
}


// Fill Bitwords
void 
FastL1GlobalAlgo::FillBitInfos() {
  m_BitInfos.clear();
  for (int i=0; i<396; i++) {
    m_BitInfos.push_back(m_Regions[i].getBitInfo());
  }
}

// ------------ Check if jet is taujet ------------
bool 
FastL1GlobalAlgo::isTauJet(int cRgn) {

  // Barrel and Endcap only
  if ((cRgn%22)<4 || (cRgn%22)>17)  
    return false;

  if (m_Regions[cRgn].GetTauBit()) 
    return false;

  int nwid = m_Regions[cRgn].GetNWId();
  int nid = m_Regions[cRgn].GetNorthId();
  int neid = m_Regions[cRgn].GetNEId();
  int wid = m_Regions[cRgn].GetWestId();
  int eid = m_Regions[cRgn].GetEastId();
  int swid = m_Regions[cRgn].GetSWId();
  int sid = m_Regions[cRgn].GetSouthId();
  int seid = m_Regions[cRgn].GetSEId();


  //Use 3x2 window at eta borders!

  // west border:
  if ((cRgn%22)==4) { 
    //std::cout << "West border check: " << std::endl
    //      << nwid << " " << nid << " "  << neid << " " << std::endl
    //      << wid << " " << cRgn << " "  << eid << " " << std::endl
    //      << swid << " " << sid << " "  << seid << " " << std::endl;    
    //std::cout << "West border check: " << std::endl
    //      << m_Regions[nwid].GetTauBit() << " " << m_Regions[nid].GetTauBit() << " "  << m_Regions[neid].GetTauBit() << " " << std::endl
    //      << m_Regions[wid].GetTauBit() << " " << m_Regions[cRgn].GetTauBit() << " "  << m_Regions[eid].GetTauBit() << " " << std::endl
    //      << m_Regions[swid].GetTauBit() << " " << m_Regions[sid].GetTauBit() << " "  << m_Regions[seid].GetTauBit() << " " << std::endl;    

    if (
	m_Regions[nid].GetTauBit()  ||
	m_Regions[neid].GetTauBit() ||
	m_Regions[eid].GetTauBit()  ||
	m_Regions[seid].GetTauBit() ||
	m_Regions[sid].GetTauBit()  ||
	m_Regions[cRgn].GetTauBit()
	) 
      return false;
    else return true;
  }
  // east border:
  if ((cRgn%22)==17) { 
    //std::cout << "East border check2: " << std::endl
    //      << nwid << " " << nid << " "  << neid << " " << std::endl
    //      << wid << " " << cRgn << " "  << eid << " " << std::endl
    //      << swid << " " << sid << " "  << seid << " " << std::endl;    
    //std::cout << "East border check: " << std::endl
    //      << m_Regions[nwid].GetTauBit() << " " << m_Regions[nid].GetTauBit() << " "  << m_Regions[neid].GetTauBit() << " " << std::endl
    //      << m_Regions[wid].GetTauBit() << " " << m_Regions[cRgn].GetTauBit() << " "  << m_Regions[eid].GetTauBit() << " " << std::endl
    //      << m_Regions[swid].GetTauBit() << " " << m_Regions[sid].GetTauBit() << " "  << m_Regions[seid].GetTauBit() << " " << std::endl;    

     if (
	m_Regions[nid].GetTauBit()  ||
	m_Regions[nwid].GetTauBit() ||
	m_Regions[wid].GetTauBit()  ||
	m_Regions[swid].GetTauBit() ||
	m_Regions[sid].GetTauBit()  ||
	m_Regions[cRgn].GetTauBit()
	) 
      return false;
    else return true;
  }

  if (nwid==999 || neid==999 || nid==999 || swid==999 || seid==999 || sid==999 || wid==999 || 
      eid==999 ) { 
    return false;
  }

/*
  if (
      m_Regions[nwid].GetTauBit() ||
      m_Regions[nid].GetTauBit()  ||
      m_Regions[neid].GetTauBit() ||
      m_Regions[wid].GetTauBit()  ||
      m_Regions[eid].GetTauBit()  ||
      m_Regions[swid].GetTauBit() ||
      m_Regions[seid].GetTauBit() ||
      m_Regions[sid].GetTauBit()
      ) 
    m_Regions[cRgn].BitInfo.setIsolationVeto ( true);
*/

  if (
      m_Regions[nwid].GetTauBit() ||
      m_Regions[nid].GetTauBit()  ||
      m_Regions[neid].GetTauBit() ||
      m_Regions[wid].GetTauBit()  ||
      m_Regions[eid].GetTauBit()  ||
      m_Regions[swid].GetTauBit() ||
      m_Regions[seid].GetTauBit() ||
      m_Regions[sid].GetTauBit()  ||
      m_Regions[cRgn].GetTauBit()
      ) 
    return false;
  else return true;


}

// ------------ Check if tower is emcand ------------
// returns 1 = non-iso EM, 2 = iso EM, 0 = no EM
int
FastL1GlobalAlgo::isEMCand(CaloTowerDetId cid, l1extra::L1EmParticle* ph,const edm::Event& iEvent) {

  // center tower
  int crgn = m_RMap->getRegionIndex(cid);
  int ctwr = m_RMap->getRegionTowerIndex(cid);

  // crystals only in barrel/endcap part
  if ((crgn%22)<4 || (crgn%22)>17) return 0;
  if (crgn>395 || crgn < 0 || ctwr > 15 || ctwr < 0) return 0;

  CaloTowerCollection c = m_Regions.at(crgn).GetCaloTowers();
  double cenEt = c[ctwr].emEt();
  //double cenE = c[ctwr].emEnergy();
  
  // Using region position rather than tower position
  std::pair<double, double> crpos = m_RMap->getRegionCenterEtaPhi(crgn);
  //double cenEta = c[ctwr].eta();
  //double cenPhi = c[ctwr].phi();
  double cenEta = crpos.first;
  double cenPhi = crpos.second;

  double cenFGbit = m_Regions.at(crgn).GetFGBit(ctwr);
  double cenHOEbit = m_Regions.at(crgn).GetHOEBit(ctwr);

  if (cenEt<m_L1Config.TowerEBThreshold) return 0;

  // check fine grain bit
  if (cenFGbit) return 0;

  // check H/E bit
  if (cenHOEbit) return 0;

  // check neighbours
  std::pair<int, int> no = m_RMap->GetTowerNorthEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> so = m_RMap->GetTowerSouthEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> we = m_RMap->GetTowerWestEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> ea = m_RMap->GetTowerEastEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> nw = m_RMap->GetTowerNWEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> ne = m_RMap->GetTowerNEEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> sw = m_RMap->GetTowerSWEtaPhi(cid.ieta(),cid.iphi()); 
  std::pair<int, int> se = m_RMap->GetTowerSEEtaPhi(cid.ieta(),cid.iphi()); 
  if (no.first>28 || no.first<-28 || no.second>72 || no.second<0) return 0;
  if (so.first>28 || so.first<-28 || so.second>72 || so.second<0) return 0;
  if (we.first>28 || we.first<-28 || we.second>72 || we.second<0) return 0;
  if (ea.first>28 || ea.first<-28 || ea.second>72 || ea.second<0) return 0;
  if (nw.first>28 || nw.first<-28 || nw.second>72 || nw.second<0) return 0;
  if (ne.first>28 || ne.first<-28 || ne.second>72 || ne.second<0) return 0;
  if (sw.first>28 || sw.first<-28 || sw.second>72 || sw.second<0) return 0;
  if (se.first>28 || se.first<-28 || se.second>72 || se.second<0) return 0;

  int notwr = m_RMap->getRegionTowerIndex(no);
  int norgn = m_RMap->getRegionIndex(no.first,no.second);
  int sotwr = m_RMap->getRegionTowerIndex(so);
  int sorgn = m_RMap->getRegionIndex(so.first,so.second);
  int wetwr = m_RMap->getRegionTowerIndex(we);
  int wergn = m_RMap->getRegionIndex(we.first,we.second);
  int eatwr = m_RMap->getRegionTowerIndex(ea);
  int eargn = m_RMap->getRegionIndex(ea.first,ea.second);
  int setwr = m_RMap->getRegionTowerIndex(se);
  int sergn = m_RMap->getRegionIndex(se.first,sw.second);
  int swtwr = m_RMap->getRegionTowerIndex(sw);
  int swrgn = m_RMap->getRegionIndex(sw.first,sw.second);
  int netwr = m_RMap->getRegionTowerIndex(ne);
  int nergn = m_RMap->getRegionIndex(ne.first,ne.second);
  int nwtwr = m_RMap->getRegionTowerIndex(nw);
  int nwrgn = m_RMap->getRegionIndex(nw.first,nw.second);
  
  //
  if (norgn>395 || norgn < 0 || notwr > 15 || notwr < 0) return 0;
  c = m_Regions[norgn].GetCaloTowers();
  double noEt = c[notwr].emEt();
  //double noE = c[notwr].emEnergy();
  // check fine grain bit
  bool noFGbit = m_Regions[norgn].GetFGBit(notwr);
  // check H/E bit
  bool noHOEbit = m_Regions[norgn].GetHOEBit(notwr);

  //
  if (sorgn>395 || sorgn < 0 || sotwr > 15 || sotwr < 0) return 0;
  c = m_Regions[sorgn].GetCaloTowers();
  double soEt = c[sotwr].emEt();
  //double soE = c[sotwr].emEnergy();
  // check fine grain bit
  bool soFGbit = m_Regions[sorgn].GetFGBit(sotwr);
  // check H/E bit
  bool soHOEbit = m_Regions[sorgn].GetHOEBit(sotwr);

  //
  if (wergn>395 || wergn < 0 || wetwr > 15 || wetwr < 0) return 0;
  c = m_Regions[wergn].GetCaloTowers();
  double weEt = c[wetwr].emEt();
  //double weE = c[wetwr].emEnergy();
  // check fine grain bit
  bool weFGbit = m_Regions[wergn].GetFGBit(wetwr);
  // check H/E bit
  bool weHOEbit = m_Regions[wergn].GetHOEBit(wetwr);

  //
  if (eargn>395 || eargn < 0 || eatwr > 15 || eatwr < 0) return 0;
  c = m_Regions[eargn].GetCaloTowers();
  double eaEt = c[eatwr].emEt();
  //double eaE = c[eatwr].emEnergy();
  // check fine grain bit
  bool eaFGbit = m_Regions[eargn].GetFGBit(eatwr);
  // check H/E bit
  bool eaHOEbit = m_Regions[eargn].GetHOEBit(eatwr);

  //
  if (nwrgn>395 || nwrgn < 0 || nwtwr > 15 || nwtwr < 0) return 0;
  c = m_Regions[nwrgn].GetCaloTowers();
  double nwEt = c[nwtwr].emEt();
  //double nwE = c[nwtwr].emEnergy();
  // check fine grain bit
  bool nwFGbit = m_Regions[nwrgn].GetFGBit(nwtwr);
  // check H/E bit
  bool nwHOEbit = m_Regions[nwrgn].GetHOEBit(nwtwr);

  //
  if (nergn>395 || nergn < 0 || netwr > 15 || netwr < 0) return 0;
  c = m_Regions[nergn].GetCaloTowers();
  double neEt = c[netwr].emEt();
  //double neE = c[netwr].emEnergy();
  // check fine grain bit
  bool neFGbit = m_Regions[nergn].GetFGBit(netwr);
  // check H/E bit
  bool neHOEbit = m_Regions[nergn].GetHOEBit(netwr);

  //
  if (swrgn>395 || swrgn < 0 || swtwr > 15 || swtwr < 0) return 0;
  c = m_Regions[swrgn].GetCaloTowers();
  double swEt = c[swtwr].emEt();
  //double swE = c[swtwr].emEnergy();
  // check fine grain bit
  bool swFGbit = m_Regions[swrgn].GetFGBit(swtwr);
  // check H/E bit
  bool swHOEbit = m_Regions[swrgn].GetHOEBit(swtwr);

  //
  if (sergn>395 || sergn < 0 || setwr > 15 || setwr < 0) return 0;
  c = m_Regions[sergn].GetCaloTowers();
  double seEt = c[setwr].emEt();
  //double seE = c[setwr].emEnergy();
  // check fine grain bit
  bool seFGbit = m_Regions[sergn].GetFGBit(setwr);
  // check H/E bit
  bool seHOEbit = m_Regions[sergn].GetHOEBit(setwr);

  
  // check if highest et tower
  bool isHit = false;
  if ( cenEt > noEt && cenEt > soEt && cenEt > weEt &&
       cenEt > eaEt && cenEt > nwEt && cenEt > neEt &&
       cenEt > swEt && cenEt > seEt ) isHit = true;
  else 
    return 0;

  // find highest neighbour
  double hitEt = cenEt;
  //double hitE = cenE;
  double maxEt = std::max(noEt,std::max(soEt,std::max(weEt,eaEt)));
  //double maxE = std::max(noE,std::max(soE,std::max(weE,eaE)));

  // check 2 tower Et
  float emEtThres = m_L1Config.EMSeedEnThreshold;
  
  // at this point candidate is at least non-iso Egamma
  //double eme = (hitE+maxE);
  //double eme = (hitE+maxE);
  double emet = (hitEt+maxEt);

  /*
  if (m_L1Config.DoEMCorr) {
    emet = GCTEnergyTrunc(corrEmEt(emet,cenEta),m_L1Config.EMLSB, true);
    //emet = GCTEnergyTrunc(emet,m_L1Config.EMLSB, true);
  } else {
    emet = GCTEnergyTrunc(emet,m_L1Config.EMLSB, true);
  }
  */
  emet = GCTEnergyTrunc(emet,m_L1Config.EMLSB, true);

  if ((emet)<emEtThres) return 0;

  double emtheta = 2.*atan(exp(-cenEta));
  //double emet = eme*sin(emtheta);
  double emex = emet*cos(cenPhi);
  double emey = emet*sin(cenPhi);
  //double emex = eme*sin(emtheta)*cos(cenPhi);
  //double emey = eme*sin(emtheta)*sin(cenPhi);
  double eme = emex/sin(emtheta)/cos(cenPhi);
  double emez = eme*cos(emtheta);


  reco::Particle::LorentzVector rp4(emex,emey,emez,eme);
  //reco::Particle::Point rp3(0.,0.,0.);
  //reco::Particle::Charge q(0);
  //*ph = reco::Photon(q,rp4,rp3);
  *ph = l1extra::L1EmParticle(rp4);
 
  //if (emet>0.) {
  //  std::cout << "em region et, eta, phi: "<< emet<<" "<< cenEta<<" "<< cenPhi<<" " << std::endl;
  //  std::cout << "em lv et, eta, phi: "<< ph->et()<<" "<< ph->eta()<<" "<< ph->phi()<<" " << std::endl;
  //}

  // check isolation FG bits
  if (noFGbit || soFGbit || weFGbit || eaFGbit || 
      nwFGbit || neFGbit || swFGbit || seFGbit ||
      noHOEbit || soHOEbit || weHOEbit || eaHOEbit || 
      nwHOEbit || neHOEbit || swHOEbit || seHOEbit)
    return 1;
  
  // check isolation corners
  //double corThres = 0.4;
  //double quietThres = m_L1Config.EMNoiseLevel;
  double quietThres = m_L1Config.QuietRegionThreshold;
  bool isoVeto1 = false,isoVeto2 = false,isoVeto3 = false,isoVeto4 = false;
  if (swEt>quietThres || weEt>quietThres || nwEt>quietThres || noEt>quietThres || neEt>quietThres ) {
    //if ((swEt + weEt + nwEt + noEt + neEt)/cenEt > corThres) 
    isoVeto1 = true;
  }
  if (neEt>quietThres || eaEt>quietThres || seEt>quietThres || soEt>quietThres || swEt>quietThres ) {
    //if ((neEt + eaEt + seEt + soEt + swEt)/cenEt > corThres) 
    isoVeto2 = true;
  }
  if (nwEt>quietThres || noEt>quietThres || neEt>quietThres || eaEt>quietThres || seEt>quietThres ) {
    //if ((nwEt + noEt + neEt + eaEt + seEt)/cenEt > corThres) 
    isoVeto3 = true;
  }
  if (seEt>quietThres || soEt>quietThres || swEt>quietThres || weEt>quietThres || nwEt>quietThres ) {
    //if ((seEt + soEt + swEt + weEt + nwEt)/cenEt > corThres) 
    isoVeto4 = true;
  }
  if (isoVeto1 && isoVeto2 && isoVeto3 && isoVeto4)
    return 1;
  
  return 2;    
}


// is central region the highest Et Region?
bool 
FastL1GlobalAlgo::isMaxEtRgn_Window33(int crgn) {

  int nwid = m_Regions.at(crgn).GetNWId();
  int nid = m_Regions.at(crgn).GetNorthId();
  int neid = m_Regions.at(crgn).GetNEId();
  int wid = m_Regions.at(crgn).GetWestId();
  int eid = m_Regions.at(crgn).GetEastId();
  int swid = m_Regions.at(crgn).GetSWId();
  int sid = m_Regions.at(crgn).GetSouthId();
  int seid = m_Regions.at(crgn).GetSEId();


  //Use 3x2 window at eta borders!
  // east border:
  if ((crgn%22)==17) { 
    
    if (nwid==999 || nid==999 || swid==999 || sid==999 || wid==999 ) { 
      return false;
    }

    double cenet = m_Regions.at(crgn).SumEt();
    double nwet =  m_Regions[nwid].SumEt();
    double noet = m_Regions[nid].SumEt();
    double weet = m_Regions[wid].SumEt();
    double swet = m_Regions[swid].SumEt();
    double soet = m_Regions[sid].SumEt();
    
    if ( cenet > nwet &&  cenet > noet &&
	 cenet > weet &&  cenet > soet &&
	 cenet > swet ) 
      {

	double cene = m_Regions.at(crgn).SumE();
	double nwe =  m_Regions[nwid].SumE();
	double noe = m_Regions[nid].SumE();
	double wee = m_Regions[wid].SumE();
	double swe = m_Regions[swid].SumE();
	double soe = m_Regions[sid].SumE();
	
	// if region is central: jet energy is sum of 3x3 region
      // surrounding the central region
	double jE = cene + nwe + noe + wee + swe + soe;
	double jEt = cenet + nwet + noet + weet + swet + soet;
	

	m_Regions.at(crgn).SetJetE(jE);
	m_Regions.at(crgn).SetJetEt(jEt);
	
	m_Regions.at(crgn).SetJetE3x3(cene);
	m_Regions.at(crgn).SetJetEt3x3(cenet);
	
	return true;
      } else { return false; }
    
  }


  // west border:
  if ((crgn%22)==4) { 
    
    if (neid==999 || nid==999 || seid==999 || sid==999 || eid==999 ) { 
      return false;
    }

    double cenet = m_Regions.at(crgn).SumEt();
    double neet =  m_Regions[neid].SumEt();
    double noet = m_Regions[nid].SumEt();
    double eaet = m_Regions[eid].SumEt();
    double seet = m_Regions[seid].SumEt();
    double soet = m_Regions[sid].SumEt();
    
    if ( cenet > neet &&  cenet > noet &&
	 cenet > eaet &&  cenet > soet &&
	 cenet > seet ) 
      {

	double cene = m_Regions.at(crgn).SumE();
	double nee =  m_Regions[neid].SumE();
	double noe = m_Regions[nid].SumE();
	double eae = m_Regions[eid].SumE();
	double see = m_Regions[seid].SumE();
	double soe = m_Regions[sid].SumE();
	
	// if region is central: jet energy is sum of 3x3 region
      // surrounding the central region
	double jE = cene + nee + noe + eae + see + soe;
	double jEt = cenet + neet + noet + eaet + seet + soet;
	
	m_Regions.at(crgn).SetJetE(jE);
	m_Regions.at(crgn).SetJetEt(jEt);
	
	m_Regions.at(crgn).SetJetE3x3(cene);
	m_Regions.at(crgn).SetJetEt3x3(cenet);
	
	return true;
      } else { return false; }
    
  }


  if (nwid==999 || neid==999 || nid==999 || swid==999 || seid==999 || sid==999 || wid==999 || 
      eid==999 ) { 
    //std::cerr << "FastL1GlobalAlgo::isMaxEtRgn_Window33(): RegionId out of bounds: " << std::endl
    //      << nwid << " " << nid << " "  << neid << " " << std::endl
    //      << wid << " " << crgn << " "  << eid << " " << std::endl
    //      << swid << " " << sid << " "  << seid << " " << std::endl;    
    return false;
  }

  double cenet = m_Regions.at(crgn).SumEt();
  double nwet =  m_Regions[nwid].SumEt();
  double noet = m_Regions[nid].SumEt();
  double neet = m_Regions[neid].SumEt();
  double weet = m_Regions[wid].SumEt();
  double eaet = m_Regions[eid].SumEt();
  double swet = m_Regions[swid].SumEt();
  double soet = m_Regions[sid].SumEt();
  double seet = m_Regions[seid].SumEt();

  if ( cenet > nwet &&  cenet > noet &&
       cenet > neet &&  cenet > eaet &&
       cenet > weet &&  cenet > soet &&
       cenet > swet &&  cenet > seet ) 
    {

      double cene = m_Regions.at(crgn).SumE();
      double nwe =  m_Regions[nwid].SumE();
      double noe = m_Regions[nid].SumE();
      double nee = m_Regions[neid].SumE();
      double wee = m_Regions[wid].SumE();
      double eae = m_Regions[eid].SumE();
      double swe = m_Regions[swid].SumE();
      double soe = m_Regions[sid].SumE();
      double see = m_Regions[seid].SumE();

      // if region is central: jet energy is sum of 3x3 region
      // surrounding the central region
      double jE = cene + nwe + noe + nee + wee + eae + swe + soe + see;
      double jEt = cenet + nwet + noet + neet + weet + eaet + swet + soet + seet;


      m_Regions.at(crgn).SetJetE(jE);
      m_Regions.at(crgn).SetJetEt(jEt);

      m_Regions.at(crgn).SetJetE3x3(cene);
      m_Regions.at(crgn).SetJetEt3x3(cenet);
      
      return true;
    } else { return false; }

}


void
FastL1GlobalAlgo::checkMapping() {

  // loop over towers ieta,iphi
  for (int j=1;j<=72;j++) {
    for (int i=-28; i<=28; i++) {
      if (i==0) continue;
      int iRgn =  m_RMap->getRegionIndex(i,j);
      std::pair<double, double> RgnEtaPhi = m_RMap->getRegionCenterEtaPhi(iRgn);
      //int iTwr = m_RMap->getRegionTowerIndex(i,j);
      std::pair<int, int> iRgnEtaPhi = m_RMap->getRegionEtaPhiIndex(iRgn);
      std::pair<int, int> iRgnEtaPhi2 = m_RMap->getRegionEtaPhiIndex(std::pair<int, int>(i,j));

      std::cout<<"---------------------------------------------------------------------------"<<std::endl;
      std::cout<<"Region:   "<<iRgn<<" | "<<RgnEtaPhi.first<<", "<<RgnEtaPhi.second*180./3.141<<std::endl;
      std::cout<<"   -      "<<iRgnEtaPhi.first<<", "<<iRgnEtaPhi.second<<std::endl;
      std::cout<<"   -      "<<iRgnEtaPhi2.first<<", "<<iRgnEtaPhi2.second<<std::endl;
      std::cout<<" Tower:   "<<i<<", "<<m_RMap->convertFromECal_to_HCal_iphi(j)<<std::endl;
      std::cout<<" TowerId: "<<m_RMap->getRegionTowerIndex(i,j)<<std::endl;

    }
  }

}

double
FastL1GlobalAlgo::hcaletValue(const int ieta,const int compET) {
  double etvalue = m_hcaluncomp[ieta][compET];//*cos(eta_ave);
  return(etvalue);
}
