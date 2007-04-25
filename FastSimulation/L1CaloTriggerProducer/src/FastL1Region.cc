// -*- C++ -*-
//
// Package:    FastL1CaloSim
// Class:      FastL1Region
// 
/**\class FastL1Region

 Description: Container class for L1 regions.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Feb 19 13:25:24 CST 2007
// $Id: FastL1Region.cc,v 1.3 2007/04/23 15:48:30 chinhan Exp $
//

#include "FastSimulation/L1CaloTriggerProducer/interface/FastL1Region.h"

FastL1Region::FastL1Region() 
{
  Towers = CaloTowerCollection(16);

  jetE = 0.;
  jetEt = 0.;

  id = 999;
  ieta = 999;
  iphi = 999;

  tauBit = false;
  quietBit = false;
  mipBit = false;
  for(int i=0;i<16;i++) {
    hcfgBit[i] = false;
    fgBit[i] = false;
    hOeBit[i] = false;
    for (int j=0;j<25;j++) {
      EMCrystalEnergy[i][j] = 0. ; // 16x25 Crystals
    }
  }

  /*
  BitInfo.eta = -9999.;
  BitInfo.phi = -9999.;
  BitInfo.TauVeto = false;
  BitInfo.EmTauVeto = false;
  BitInfo.HadTauVeto = false;
  BitInfo.SumEtBelowThres = false;
  BitInfo.IsolationVeto = false;
  */

  // default values
  Config.EMSeedEnThreshold = 2.;
  Config.EMActiveLevel = 3.;
  Config.HadActiveLevel = 3.;
  Config.noTauVetoLevel = 10000.;	
  Config.hOeThreshold = 0.05;
  Config.FGEBThreshold = 0.8;
  Config.noFGThreshold = 50.;
  Config.FGEEThreshold = 0.8;
  Config.MuonNoiseLevel = 2.;
  Config.EMNoiseLevel = 2.;
  Config.HadNoiseLevel = 2.;
  Config.QuietRegionThreshold = 2.;  
  Config.JetSeedEtThreshold = 2.;  
  Config.CrystalEBThreshold = 0.09;
  Config.CrystalEEThreshold = 0.45;

  Config.TowerEMLSB = 1.;
  Config.TowerHadLSB = 1.;
  Config.JetLSB = 1.;

  Config.TowerEBThreshold = 0.2;
  Config.TowerEEThreshold = 0.45;
  Config.TowerHBThreshold = 0.9;
  Config.TowerHEThreshold = 1.4;

  Config.TowerEBScale = 1.0;
  Config.TowerEEScale = 1.0;
  Config.TowerHBScale = 1.0;
  Config.TowerHEScale = 1.0;

  //Config.EmInputs;
  //Config.xTowerInput;

}


FastL1Region::~FastL1Region() 
{
}


void
FastL1Region::SetParameters(FastL1Config iconfig) 
{ 
  Config = iconfig;
}

void 
FastL1Region::SetRegionEnergy()
{
  sumE = CalcSumE();
  sumEt = CalcSumEt();
}

void 
FastL1Region::SetRegionBits(edm::Event const& e)
{
  SetTauBit(e);
  SetQuietBit();
  SetMIPBit();
}

void 
FastL1Region::SetTowerBits()
{
  SetFGBit();
  SetHOEBit();
  SetHCFGBit();
}

//
void
FastL1Region::FillEMCrystals(const edm::Event& e, const edm::EventSetup& c,FastL1RegionMap* m_RMap) 
{
  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  c.get<IdealGeometryRecord>().get(cttopo);
  const CaloTowerConstituentsMap* theTowerConstituentsMap = cttopo.product();

  edm::ESHandle<CaloTopology> calotopo;
  c.get<CaloTopologyRecord>().get(calotopo);

  edm::ESHandle<CaloGeometry> cGeom; 
  c.get<IdealGeometryRecord>().get(cGeom);    

  edm::Handle<EcalRecHitCollection> ec;


  //std::vector< std::pair <std::string,std::string> > la;
  //la.resize(2);
  //// Barrel
  //la[0].first = "ecalRecHit";
  //la[0].second = "EcalRecHitsEB";
  //// EndCap
  //la[1].first = "ecalRecHit";
  //la[1].second = "EcalRecHitsEE";


  double ethres = Config.CrystalEBThreshold;

  // EB
  //e.getByLabel(la[0].first,la[0].second,ec);
  e.getByLabel(Config.EmInputs.at(0),ec);

  ethres = Config.CrystalEBThreshold;
  for(EcalRecHitCollection::const_iterator ecItr = ec->begin();
      ecItr != ec->end(); ++ecItr) {
    CaloRecHit recHit = (CaloRecHit)(*ecItr);
    if (recHit.energy()<ethres) continue;

    EBDetId detId = recHit.detid();

    //const GlobalPoint gP = cGeom->getPosition(detId);

    //CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
    // loop over towers
    for(int i=0;i<16;i++) {
      //int hiphi = m_RMap->convertFromECal_to_HCal_iphi(detId.tower_iphi());
      //int hiphi = m_RMap->convertFromHCal_to_ECal_iphi(detId.tower_iphi());
      int hiphi = detId.tower_iphi();
      if (Towers[i].id().iphi()==hiphi && 
	  Towers[i].id().ieta()==detId.tower_ieta() ) {
	int crIeta = 999;
	if (detId.ieta()>0) crIeta = (detId.ieta()-1)%5;
	else crIeta = 4 + (detId.ieta()+1)%5;
	int crIphi = (detId.iphi() - 1)%5;
	
	EMCrystalEnergy[i][crIeta + 5*crIphi] = recHit.energy();
      }
    }  
  }
  
  // After having filled crsystal info set all veto bits
  SetTowerBits();
 
  // EE FG bits are filled here!!!
  //e.getByLabel(la[1].first,la[1].second,ec);

  if (GetiEta()==4 || GetiEta()==5 ||  GetiEta()==6 ||  
      GetiEta()==15 || GetiEta()==16 || GetiEta()==17 ) {
    
    e.getByLabel(Config.EmInputs.at(1),ec);
    ethres = Config.CrystalEEThreshold;
    double towerEnergy[16];
    // loop over towers
    for(int i=0;i<16;i++) {
      fgBit[i] = false; // re-iniate
            
      //if (Towers[i].hadEt()>Config.HadNoiseLevel && Towers[i].emEt()>Config.EMNoiseLevel ) {
      if (Towers[i].emEt()>=Config.EMNoiseLevel ) {
      //if (Towers[i].emEnergy()>Config.EMNoiseLevel ) {
	//towerEnergy[i]  = Towers[i].hadEt() + Towers[i].emEt(); 
	towerEnergy[i]  = Towers[i].hadEnergy() + Towers[i].emEnergy(); 
      } else {
	fgBit[i] = false;
	continue;
      }

      // EB/EE transition area: unset fg bits
      // if (std::abs(Towers[i].id().ieta())==16 || std::abs(Towers[i].id().ieta())==17) {
      // fgBit[i] = false;
      // continue;
      // }
      if (Towers[i].emEt()>Config.noFGThreshold) {
	fgBit[i] = false;
	continue;
      }
      
      CaloRecHit maxRecHit;
      CaloRecHit maxRecHit2;
      double max2En = 0.;
      
      for(EcalRecHitCollection::const_iterator ecItr = ec->begin();
	  ecItr != ec->end(); ++ecItr) {
	CaloRecHit recHit = (CaloRecHit)(*ecItr);
	if (recHit.energy()<ethres) continue;
	
	EEDetId detId = recHit.detid();
	
	CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
	//int hiphi = m_RMap->convertFromECal_to_HCal_iphi(towerDetId.iphi());
	int hiphi = towerDetId.iphi();
	if (Towers[i].id().iphi()==hiphi && 
	    Towers[i].id().ieta()==towerDetId.ieta() ) {	
	  if (maxRecHit.energy()<recHit.energy()) {
	    maxRecHit = recHit;
	  }	
	}
      } 

      std::vector<DetId> westV = calotopo->west(maxRecHit.detid());
      std::vector<DetId> eastV = calotopo->east(maxRecHit.detid());
      std::vector<DetId> southV = calotopo->south(maxRecHit.detid());
      std::vector<DetId> northV = calotopo->north(maxRecHit.detid());
      for(EcalRecHitCollection::const_iterator ecItr = ec->begin();
	  ecItr != ec->end(); ++ecItr) {
	CaloRecHit recHit = (CaloRecHit)(*ecItr);
	if (recHit.energy()<ethres) continue;
	
	EEDetId detId = recHit.detid();
	
	CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
	//int hiphi = m_RMap->convertFromECal_to_HCal_iphi(towerDetId.iphi());
	int hiphi = towerDetId.iphi();
	if (Towers[i].id().iphi()==hiphi && 
	    Towers[i].id().ieta()==towerDetId.ieta() ) {
	  if ( 
	      (!westV.empty() && recHit.detid()==westV[0]) || 
	      (!eastV.empty() && recHit.detid()==eastV[0]) || 
	      (!northV.empty() && recHit.detid()==northV[0]) || 
	      (!southV.empty() && recHit.detid()==southV[0]) 
	      ) {
	    if (maxRecHit2.energy()<recHit.energy()) {
	      maxRecHit2 = recHit;
	    }	
	    max2En += recHit.energy();
	  }
	}
      }  
      
      double eeThres = Config.FGEEThreshold;
      //double totE = maxRecHit.energy() + max2En;
      double totE = maxRecHit.energy() + maxRecHit2.energy();
      if (towerEnergy[i]>(Config.TowerEEThreshold)) {
	//double totE = maxRecHit.energy() + maxRecHit2.energy();
	//if (totE/towerEnergy[i]<Config.FGEBThreshold) fgBit[i] = true;    
	if (totE/towerEnergy[i]<eeThres) fgBit[i] = true;    
      }
            
    }
  }
   
 
}

//
void
FastL1Region::FillTowerZero(const CaloTower& t, int& tid) 
{
  Towers[tid] = CaloTower(t);
}

void
FastL1Region::FillTower(const CaloTower& t, int& tid) 
{
  double EThres = 0.;
  double HThres = 0.;
  double EBthres = Config.TowerEBThreshold;
  double HBthres = Config.TowerHBThreshold;
  double EEthres = Config.TowerEBThreshold;
  double HEthres = Config.TowerEEThreshold;
  
  if(std::abs(t.eta())<2.322) {
    EThres = EBthres;
  } else {
    EThres = EEthres;
  }
  if(std::abs(t.eta())<2.322) {
    HThres = HBthres;
  } else {
    HThres = HEthres;
  }

  double emet = TPEnergyRound(emet,Config.TowerEMLSB,EThres);
  double hadet = TPEnergyRound(hadet,Config.TowerHadLSB,HThres);
  double eme = TPEnergyRound(eme,Config.TowerEMLSB,EThres);
  double hade = TPEnergyRound(hade,Config.TowerHadLSB,HThres);

  if ( emet<EThres) emet = 0.;
  if ( hadet<HThres) hadet = 0.;
  //if ( eme<EThres) emet = 0.;
  //if ( hade<HThres) hadet = 0.;
 
  //Towers[tid] = CaloTower(t);
  Towers[tid] = CaloTower(t.id(),t.momentum(),emet,hadet,t.outerEt(),0,0);
}


//
void
FastL1Region::FillTower_Scaled(const CaloTower& t, int& tid) 
{
  double EThres = 0.;
  double HThres = 0.;
  double EBthres = Config.TowerEBThreshold;
  double HBthres = Config.TowerHBThreshold;
  double EEthres = Config.TowerEBThreshold;
  double HEthres = Config.TowerEEThreshold;
  
  if(std::abs(t.eta())<2.322) {
    EThres = EBthres;
  } else {
    EThres = EEthres;
  }
  if(std::abs(t.eta())<2.322) {
    HThres = HBthres;
  } else {
    HThres = HEthres;
  }

  double emScale = 1.0;
  double hadScale = 1.0;
  //double outerScale = 1.0;

  if (std::abs(t.eta()>1.3050) && std::abs(t.eta())<3.0) {
    hadScale = Config.TowerHEScale;
    emScale = Config.TowerEEScale;
  }
  if (std::abs(t.eta()<1.3050)) {
    hadScale = Config.TowerHBScale;
    emScale = Config.TowerEBScale;
  }

  double emet = emScale * t.emEt();
  double hadet = hadScale * t.hadEt();
  double eme = emScale * t.emEnergy();
  double hade = hadScale * t.hadEnergy();

  emet = TPEnergyRound(emet,Config.TowerEMLSB,EThres);
  hadet = TPEnergyRound(hadet,Config.TowerHadLSB,HThres);
  eme = TPEnergyRound(eme,Config.TowerEMLSB,EThres);
  hade = TPEnergyRound(hade,Config.TowerHadLSB,HThres);

  if ( emet<EThres) emet = 0.;
  if ( hadet<HThres) hadet = 0.;
  //if ( eme<EThres) emet = 0.;
  //if ( hade<HThres) hadet = 0.;
 
  //Towers[tid] = CaloTower(t);
  Towers[tid] = CaloTower(t.id(),t.momentum(),emet,hadet,t.outerEt(),0,0);
  
}

void 
FastL1Region::SetHOEBit()
{
  double fracThres = Config.hOeThreshold;

  for (int i=0; i<16; i++) {
    //if (Towers[i].hadEt()>Config.HadNoiseLevel && Towers[i].emEt()>Config.EMNoiseLevel ) {
    if (Towers[i].hadEnergy()>Config.HadNoiseLevel && Towers[i].emEnergy()>Config.EMNoiseLevel ) {
      if((Towers[i].hadEt()/Towers[i].emEt()) > fracThres) {
	hOeBit[i] = true;
      }
    }
  }
}

void 
FastL1Region::SetQuietBit()
{
  if (SumEt()<Config.QuietRegionThreshold)
    quietBit = true;
}

void 
FastL1Region::SetHCFGBit()
{
  // temporary: check definition 
  // if (Tower->hadEt>100GeV) hcfgBit = true; ????
  //for (int i=0; i<16; i++) {
  //}
}

void 
FastL1Region::SetMIPBit()
{
  if (quietBit)
  for (int i=0; i<16; i++) {
    if (hcfgBit) {
      mipBit = true;
      return;
    }
  }
}

void 
FastL1Region::SetFGBit()
{
  double ratioCut = Config.FGEBThreshold;

  double stripEnergy[16][5];
  double duostripEnergy[16][4];
  double towerEnergy[16];

  if (GetiEta()>=7 && GetiEta()<=14) {
    //Barrel
    for (int i=0; i<16; i++) {
      //if (Towers[i].hadEt()>Config.HadNoiseLevel && Towers[i].emEt()>Config.EMNoiseLevel ) {
      if (Towers[i].emEt()>Config.EMNoiseLevel ) {
	//towerEnergy[i]  = Towers[i].hadEt() + Towers[i].emEt(); 
	towerEnergy[i]  = Towers[i].hadEnergy() + Towers[i].emEnergy(); 
      } else {
	fgBit[i] = false;
	continue;
      }

      // EB/EE transition area: unset fg bits
      //if (std::abs(Towers[i].id().ieta())==16 || std::abs(Towers[i].id().ieta())==17) {
      //fgBit[i] = false;
      //continue;
      //}
      if (Towers[i].emEt()>Config.noFGThreshold) {
	fgBit[i] = false;
	continue;
      }

      bool fgflag = false;
	for (int j=0; j<5; j++) {
	stripEnergy[i][j] = EMCrystalEnergy[i][j] + EMCrystalEnergy[i][j+5] + EMCrystalEnergy[i][j+10] + 
	  EMCrystalEnergy[i][j+15] + EMCrystalEnergy[i][j+20];
      }
      for (int j=0; j<4; j++) {
	duostripEnergy[i][j] = stripEnergy[i][j] + stripEnergy[i][j+1];
	if (towerEnergy[i]>(Config.TowerEBThreshold)) {
	  //std::cout<<duostripEnergy[i][j]<<" |"<<towerEnergy[i]<<" |"<<duostripEnergy[i][j]/towerEnergy[i]<<std::endl;
	  if ( (duostripEnergy[i][j] / towerEnergy[i]) > ratioCut) {
	    fgflag = true;
	  } 
	  //std::cout<<duostripEnergy[i][j]<<" | "<<towerEnergy[i]<<": "<<duostripEnergy[i][j]/towerEnergy[i]<<std::endl;	
	}
      }

      if (fgflag) { 
	fgBit[i] = false;
      } else {
	fgBit[i] = true;
      }
      //std::cout<<GetiEta()<<" | "<<i<<": "<<fgBit[i]<<std::endl;
      //std::cout<<"********************************************"<<std::endl;
    }
  } else {
    // Endcap FG bit is already filled in fillEMCrystals()!!! 
  }
  
}


void 
FastL1Region::SetTauBit(edm::Event const& iEvent)
{
  float emThres = Config.EMActiveLevel;
  float hadThres = Config.HadActiveLevel;

  // init pattern containers
  unsigned emEtaPat = 0;
  unsigned emPhiPat = 0;
  unsigned hadEtaPat = 0;
  unsigned hadPhiPat = 0;
  unsigned one = 1;


  // fill hits as bit pattern
  for (int i=0; i<16; i++) {
    if(Towers[i].emEt() > emThres) {
      emEtaPat |= (one << (unsigned)(i%4));
      emPhiPat |= (one << (unsigned)(i/4));
    }

    if( Towers[i].hadEt() > hadThres) {
      hadEtaPat |= (one << (unsigned)(i%4));
      hadPhiPat |= (one << (unsigned)(i/4));
    }

  }

  // Patterns with two or less contiguous bits set are passed
  // rest are vetoed; 5=0101;7=0111;9=1001;10=1010;11=1011;13=1101;14=1110;15=1111
  //  --- Alternate patterns
  //  --- 9=1001;15=1111
  static std::vector<unsigned> vetoPatterns;
  if(vetoPatterns.size() == 0) {
    vetoPatterns.push_back(5);
    vetoPatterns.push_back(7);
    vetoPatterns.push_back(9);
    vetoPatterns.push_back(10);
    vetoPatterns.push_back(11);
    vetoPatterns.push_back(13);
    vetoPatterns.push_back(14);
    vetoPatterns.push_back(15);
  }


  for(std::vector<unsigned>::iterator i = vetoPatterns.begin();
      i != vetoPatterns.end();  i++) {
    unsigned etaPattern = emEtaPat | hadEtaPat;
    unsigned phiPattern = emPhiPat | hadPhiPat;

    /*
    //  em pattern
    if(emEtaPat == *i || emPhiPat == *i) {
      BitInfo.EmTauVeto = true;
    }
    //  had pattern
    if(hadEtaPat == *i || hadPhiPat == *i) {
      BitInfo.HadTauVeto = true;
    }
    */

    if(etaPattern == *i || phiPattern == *i) // combined pattern
      //if(emEtaPat == *i || emPhiPat == *i || hadEtaPat == *i || hadPhiPat == *i)
      {
	tauBit = true;
	//BitInfo.TauVeto = true;

	
	return;
      }  
    
  }

  tauBit = false;
  
}




double 
FastL1Region::CalcSumEt()
{
  double sumet=0;
  for (int i=0; i<16; i++) {
    sumet += Towers[i].emEt();
    sumet += Towers[i].hadEt();
  }

  return sumet;
}

double 
FastL1Region::CalcSumE()
{
  double sume=0;
  for (int i=0; i<16; i++) {
    sume += Towers[i].emEnergy();
    sume += Towers[i].hadEnergy();

  }
  return sume;
}


std::pair<double, double>
FastL1Region::getRegionCenterEtaPhi(const edm::EventSetup& c)
{
  edm::ESHandle<CaloGeometry> cGeom; 
  c.get<IdealGeometryRecord>().get(cGeom);    

  const GlobalPoint gP1 = cGeom->getPosition(Towers[5].id());
  //const GlobalPoint gP2 = cGeom->getPosition(Towers[6].id());
  //const GlobalPoint gP3 = cGeom->getPosition(Towers[10].id());

  double eta = gP1.eta();  
  double phi = gP1.phi();
  
  return std::pair<double, double>(eta, phi);
}


// test filling of FastL1Region
void 
FastL1Region::Dump()
{

  // test tower filling:
  /*
  CaloTowerCollection::const_iterator t;
  int count = 0;
  for (t=Towers.begin(); t!=Towers.end(); t++) {
    std::cout << count << ") " << t->energy() << " | " << t->eta()  << " | " << t->phi() << std::endl;
    count++;
  }
  std::cout << std::endl;
  */

  // test region neighbours:
  std::cout << this->GetNWId() << " "  << this->GetNorthId() << " "  << this->GetNEId() << std::endl;
  std::cout << this->GetWestId() << " "  << this->GetId() << " "  << this->GetEastId() << std::endl;
  std::cout << this->GetSWId() << " "  << this->GetSouthId() << " "  << this->GetSEId() << std::endl;
  std::cout << std::endl;

}

int 
FastL1Region::GetNorthId() 
{ if (iphi != 17) return ((iphi+1)*22 + ieta); else return ieta; }


int 
FastL1Region::GetSouthId() 
{ if (iphi != 0) return ((iphi-1)*22 + ieta); else return (17*22 + ieta); }

int 
FastL1Region::GetWestId() 
{ if (ieta != 0) return (iphi*22 + ieta-1); else return 999; }

int 
FastL1Region::GetEastId() 
{ if (ieta != 21) return (iphi*22 + ieta+1); else return 999; }
  
int 
FastL1Region::GetNWId() 
{ 
  if (ieta != 0) {
    if (iphi != 17) 
      return ((iphi+1)*22 + ieta-1); 
    else
      return (ieta-1); 	
  } else {
    return 999; 
  }
}

int 
FastL1Region::GetSWId() 
{ 
  if (ieta != 0) {
    if (iphi != 0) 
      return ((iphi-1)*22 + ieta-1); 
    else
      return (17*22 + ieta-1); 	
  } else {
    return 999; 
  }
}

int FastL1Region::GetNEId() 
{ 
  if (ieta != 21) {
    if (iphi != 17) 
      return ((iphi+1)*22 + ieta+1); 
    else
      return (ieta+1); 	
  } else {
    return 999; 
  }
}

int 
FastL1Region::GetSEId() 
{ 
  if (ieta != 21) {
    if (iphi != 0) 
      return ((iphi-1)*22 + ieta+1); 
    else
      return (17*22 + ieta+1); 	
  } else {
    return 999; 
  }
}

// Rounding the Et info for simulating the regional Et resolution
double 
TPEnergyRound(double et, double Resol = 1., double thres = 1.) {
  double ret = (int)(et / Resol) * Resol;
  //if (et>=thres) ret += Resol;
  //else ret = 0.;

  return ret;
}



