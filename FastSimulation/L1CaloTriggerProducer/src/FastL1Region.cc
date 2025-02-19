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
// $Id: FastL1Region.cc,v 1.23 2009/03/23 11:41:28 chinhan Exp $
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
  Config.EMLSB = 1.;
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
FastL1Region::SetParameters(L1Config iconfig) 
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
FastL1Region::FillEMCrystals(const CaloTowerConstituentsMap* theTowerConstituentsMap,
			     const CaloTopology* calotopo,
			     const CaloGeometry* cGeom,
			     const EcalRecHitCollection* ec0,
			     const EcalRecHitCollection* ec1,
			     FastL1RegionMap* m_RMap) 
{
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
  //e.getByLabel(Config.EmInputs.at(0),ec);

  ethres = Config.CrystalEBThreshold;
  for(EcalRecHitCollection::const_iterator ecItr = ec0->begin();
      ecItr != ec0->end(); ++ecItr) {
    //CaloRecHit recHit = (CaloRecHit)(*ecItr);
    if (ecItr->energy()<ethres) continue;

    EBDetId detId = ecItr->detid();

    //int hiphi = detId.tower_iphi();
    int hieta = detId.tower_ieta();
    int eieta = detId.ieta();
    int eiphi = detId.iphi();
    int crIeta = 999;
    if (hieta>0)
      crIeta = (eieta-1)%5;
    else
      crIeta = 4 + (eieta+1)%5;
    int crIphi = (eiphi - 1)%5;
    
    //const GlobalPoint gP = cGeom->getPosition(detId);

    //CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
    // loop over towers
    for(int i=0;i<16;i++) {
      //int hiphi = m_RMap->convertFromECal_to_HCal_iphi(detId.tower_iphi());
      //int hiphi = m_RMap->convertFromHCal_to_ECal_iphi(detId.tower_iphi());
      int hiphi = detId.tower_iphi();
      if ( !Towers[i].id().iphi()==hiphi ||  !Towers[i].id().ieta()==hieta ) continue;
      EMCrystalEnergy[i][crIeta + 5*crIphi] = ecItr->energy();
    }  
  }
  
  // After having filled crsystal info set all veto bits
  SetTowerBits();
 
  // EE FG bits are filled here!!!
  //e.getByLabel(la[1].first,la[1].second,ec);

  if (GetiEta()==4 || GetiEta()==5 ||  GetiEta()==6 ||  
      GetiEta()==15 || GetiEta()==16 || GetiEta()==17 ) {
    
    //e.getByLabel(Config.EmInputs.at(1),ec);
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
      
      //CaloRecHit maxRecHit;
      //CaloRecHit maxRecHit2;
      double maxRecHit=-1.;
      double maxRecHit2=-1.;
      DetId maxDetId;

      double max2En = 0.;
      
      for(EcalRecHitCollection::const_iterator ecItr = ec1->begin();
	  ecItr != ec1->end(); ++ecItr) {
	//CaloRecHit recHit = (CaloRecHit)(*ecItr);
	if (ecItr->energy()<ethres) continue;
	
	EEDetId detId = ecItr->detid();
	
	CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
	//int hiphi = m_RMap->convertFromECal_to_HCal_iphi(towerDetId.iphi());
	int hiphi = towerDetId.iphi();
	if (Towers[i].id().iphi()==hiphi && 
	    Towers[i].id().ieta()==towerDetId.ieta() ) {	
	  if (maxRecHit<ecItr->energy()) {
	    maxRecHit = ecItr->energy();
	    maxDetId = detId;
	  }	
	}
      } 

      std::vector<DetId> westV = calotopo->west(maxDetId);
      std::vector<DetId> eastV = calotopo->east(maxDetId);
      std::vector<DetId> southV = calotopo->south(maxDetId);
      std::vector<DetId> northV = calotopo->north(maxDetId);
      for(EcalRecHitCollection::const_iterator ecItr = ec1->begin();
	  ecItr != ec1->end(); ++ecItr) {
	//CaloRecHit recHit = (CaloRecHit)(*ecItr);
	if (ecItr->energy()<ethres) continue;
	
	EEDetId detId = ecItr->detid();
	
	CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
	//int hiphi = m_RMap->convertFromECal_to_HCal_iphi(towerDetId.iphi());
	int hiphi = towerDetId.iphi();
	if (Towers[i].id().iphi()==hiphi && 
	    Towers[i].id().ieta()==towerDetId.ieta() ) {
	  if ( 
	      (!westV.empty() && detId==westV[0]) || 
	      (!eastV.empty() && detId==eastV[0]) || 
	      (!northV.empty() && detId==northV[0]) || 
	      (!southV.empty() && detId==southV[0]) 
	      ) {
	    if (maxRecHit2<ecItr->energy()) {
	      maxRecHit2 = ecItr->energy();
	    }	
	    max2En += ecItr->energy();
	  }
	}
      }  
      
      double eeThres = Config.FGEEThreshold;
      //double totE = maxRecHit.energy() + max2En;
      double totE = maxRecHit + maxRecHit2;
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
  //std::cout<<"--- "<<Towers[tid].emEt()<<" "<<Towers[tid].hadEt()<<std::endl;
}

void
FastL1Region::FillTower(const CaloTower& t, int& tid, edm::ESHandle<CaloGeometry> &cGeom) 
{
  double EThres = 0.;
  double HThres = 0.;
  double EBthres = Config.TowerEBThreshold;
  double HBthres = Config.TowerHBThreshold;
  double EEthres = Config.TowerEEThreshold;
  double HEthres = Config.TowerHEThreshold;
  
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

  double upperThres = 1024.;
  double emet = RCTEnergyTrunc(t.emEt(),Config.TowerEMLSB,upperThres);
  double hadet = RCTEnergyTrunc(t.hadEt(),Config.TowerHadLSB,upperThres);
  //double eme = RCTEnergyTrunc(t.emEnergy(),Config.TowerEMLSB,upperThres);
  //double hade = RCTEnergyTrunc(t.hadEnergy(),Config.TowerHadLSB,upperThres);

  if ( emet<EThres) emet = 0.;
  if ( hadet<HThres) hadet = 0.;
  //if ( eme<EThres) emet = 0.;
  //if ( hade<HThres) hadet = 0.;
 
  GlobalPoint gP = cGeom->getPosition(t.id());
  math::XYZTLorentzVector lvec(t.px(),t.py(),t.px(),t.energy());
  //Towers[tid] = CaloTower(t);
  //Towers[tid] = CaloTower(t.id(),t.momentum(),emet,hadet,t.outerEt(),0,0);
  Towers[tid] = CaloTower(t.id(),emet,hadet,t.outerEt(),0,0,lvec,gP,gP);
}


//
void
FastL1Region::FillTower_Scaled(const CaloTower& t, int& tid, bool doRCTTrunc,edm::ESHandle<CaloGeometry> &cGeom) 
{

  double EThres = 0.;
  double HThres = 0.;
  double EBthres = Config.TowerEBThreshold;
  double HBthres = Config.TowerHBThreshold;
  double EEthres = Config.TowerEEThreshold;
  double HEthres = Config.TowerHEThreshold;
  
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

  if (doRCTTrunc) {
    double upperThres = 1024.;
    emet = RCTEnergyTrunc(emet,Config.TowerEMLSB,upperThres);
    hadet = RCTEnergyTrunc(hadet,Config.TowerHadLSB,upperThres);
    eme = RCTEnergyTrunc(eme,Config.TowerEMLSB,upperThres);
    hade = RCTEnergyTrunc(hade,Config.TowerHadLSB,upperThres);
  }
  if ( emet<EThres) emet = 0.;
  if ( hadet<HThres) hadet = 0.;
  //if ( eme<EThres) emet = 0.;
  //if ( hade<HThres) hadet = 0.;

  /* 
  if (t.emEt()>0. || t.hadEt()>0.) {
    std::cout<<"+++ "
	     <<t.emEt()<<" "<<t.hadEt()<<" "
	     <<t.eta()<<" "<<t.phi()<<" "
	     <<std::endl;
  }
  */

  //Towers[tid] = CaloTower(t);
  //Towers[tid] = CaloTower(t.id(),t.momentum(),emet,hadet,0.,0,0);
  //edm::ESHandle<CaloGeometry> cGeom; 
  //c.get<CaloGeometryRecord>().get(cGeom);    
  GlobalPoint gP = cGeom->getPosition(t.id());
  math::XYZTLorentzVector lvec(t.px(),t.py(),t.px(),t.energy());
  //Towers[tid] = CaloTower(t);
  //Towers[tid] = CaloTower(t.id(),t.momentum(),emet,hadet,t.outerEt(),0,0);
  Towers[tid] = CaloTower(t.id(),emet,hadet,t.outerEt(),0,0,lvec,gP,gP);
  
  //std::cout<<tid<<"  "<<Towers[tid].emEt()<< " " <<Towers[tid].hadEt()<< std::endl;

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
FastL1Region::SetFGBit(int twrid,bool FGBIT)
{
  fgBit[twrid] = FGBIT;
}
void 
FastL1Region::SetHCFGBit(int twrid,bool FGBIT)
{
  ;
}
void 
FastL1Region::SetHOEBit(int twrid,bool FGBIT)
{
  hOeBit[twrid] = FGBIT;
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

  if (doBitInfo) BitInfo.setIsolationVeto(false);	
  if (doBitInfo) BitInfo.setTauVeto(false);	
  if (doBitInfo) BitInfo.setEmTauVeto(false);	
  if (doBitInfo) BitInfo.setHadTauVeto(false);	

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

    //  em pattern
    if(emEtaPat == *i || emPhiPat == *i) {
      if (doBitInfo) BitInfo.setEmTauVeto(true);
    }
    //  had pattern
    if(hadEtaPat == *i || hadPhiPat == *i) {
      if (doBitInfo) BitInfo.setHadTauVeto(true);
    }

    if(etaPattern == *i || phiPattern == *i) // combined pattern
      //if(emEtaPat == *i || emPhiPat == *i || hadEtaPat == *i || hadPhiPat == *i)
      {
	tauBit = true;
	if (doBitInfo) BitInfo.setTauVeto(true);	
	return;
      }  
  }
  
  tauBit = false;
  
}


int 
FastL1Region::HighestEtTowerID()
{
  int hid = -1;
  double tmpet=0.;
  for (int i=0; i<16; i++) {
    if ( (Towers[i].emEt()+Towers[i].hadEt()) > tmpet) {
      tmpet = (Towers[i].emEt()+Towers[i].hadEt());
      hid = i;
    }
  }

  if (doBitInfo) BitInfo.setHighestEtTowerID (hid); 	 

  return hid;
}

int 
FastL1Region::HighestEmEtTowerID()
{
  int hid = -1;
  double tmpet=0.;
  for (int i=0; i<16; i++) {
    if ( (Towers[i].emEt()) > tmpet) {
      tmpet = (Towers[i].emEt());
      hid = i;
    }
  }

  if (doBitInfo) BitInfo.setHighestEmEtTowerID (hid); 	 
  return hid;
}

int 
FastL1Region::HighestHadEtTowerID()
{
  int hid = -1;
  double tmpet=0.;
  for (int i=0; i<16; i++) {
    if ( (Towers[i].hadEt()) > tmpet) {
      tmpet = (Towers[i].hadEt());
      hid = i;
    }
  }

  if (doBitInfo) BitInfo.setHighestHadEtTowerID (hid); 	 
  return hid;
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
FastL1Region::CalcSumEmEt()
{
  double sumet=0;
  for (int i=0; i<16; i++) {
    sumet += Towers[i].emEt();
  }

  return sumet;
}

double 
FastL1Region::CalcSumHadEt()
{
  double sumet=0;
  for (int i=0; i<16; i++) {
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

double 
FastL1Region::CalcSumEmE()
{
  double sume=0;
  for (int i=0; i<16; i++) {
    sume += Towers[i].emEnergy();
  }
  return sume;
}

double 
FastL1Region::CalcSumHadE()
{
  double sume=0;
  for (int i=0; i<16; i++) {
    sume += Towers[i].hadEnergy();
  }
  return sume;
}


std::pair<double, double>
FastL1Region::getRegionCenterEtaPhi(const edm::EventSetup& c)
{
  edm::ESHandle<CaloGeometry> cGeom; 
  //c.get<IdealGeometryRecord>().get(cGeom);    
  c.get<CaloGeometryRecord>().get(cGeom);    

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

double 
corrJetEt(double et, double eta)
{
  return corrJetEt1(et,eta);
  //return corrJetEt2(et,eta);
}


// Jet Calibration from CMSSW_1_3_0
double 
corrJetEt2(double et, double eta)
{
  const int NUMBER_ETA_VALUES = 11;
  std::vector< std::vector<float> > m_calibFunc;

  m_calibFunc.resize(NUMBER_ETA_VALUES);

  // still fill manually 
  m_calibFunc.at(0).push_back(1);
  m_calibFunc.at(0).push_back(1);
  m_calibFunc.at(0).push_back(2);

  m_calibFunc.at(1).push_back(1);
  m_calibFunc.at(1).push_back(2);
  m_calibFunc.at(1).push_back(2);

  m_calibFunc.at(2).push_back(2);
  m_calibFunc.at(2).push_back(2);
  m_calibFunc.at(2).push_back(2);
  m_calibFunc.at(2).push_back(2);
  m_calibFunc.at(2).push_back(3);
  m_calibFunc.at(2).push_back(3);

  m_calibFunc.at(3).push_back(1);
  m_calibFunc.at(3).push_back(1);
  m_calibFunc.at(3).push_back(3);

  m_calibFunc.at(4).push_back(1);
  m_calibFunc.at(4).push_back(3);
  m_calibFunc.at(4).push_back(3);
  m_calibFunc.at(4).push_back(6);
  m_calibFunc.at(4).push_back(6);
  m_calibFunc.at(4).push_back(6);
  m_calibFunc.at(4).push_back(6);
  m_calibFunc.at(4).push_back(6);

  m_calibFunc.at(5).push_back(3);
  m_calibFunc.at(5).push_back(3);
  m_calibFunc.at(5).push_back(3);

  m_calibFunc.at(6).push_back(1);
  m_calibFunc.at(6).push_back(1);
  m_calibFunc.at(6).push_back(4);

  m_calibFunc.at(7).push_back(1);
  m_calibFunc.at(7).push_back(4);
  m_calibFunc.at(7).push_back(4);

  m_calibFunc.at(8).push_back(4);
  m_calibFunc.at(8).push_back(4);
  m_calibFunc.at(8).push_back(4);
  m_calibFunc.at(8).push_back(1);
  m_calibFunc.at(8).push_back(1);
  m_calibFunc.at(8).push_back(1);

  m_calibFunc.at(9).push_back(1);
  m_calibFunc.at(9).push_back(1);
  m_calibFunc.at(9).push_back(5);

  m_calibFunc.at(10).push_back(1);
  m_calibFunc.at(10).push_back(5);
  m_calibFunc.at(10).push_back(5);


  double etabin[12] = { 0.0, 0.348, 0.696, 1.044, 1.392, 1.74, 2.172, 3.0,
			3.33, 3.839, 4.439, 5.115};
  int BinID = 0;
  for(int i = 0; i < 11; i++){
    if(std::abs(eta) >= etabin[i] && eta < etabin[i+1])
      BinID = i;
  }

  double corrEt = 0;
  for (unsigned i=0; i<m_calibFunc.at(BinID).size();i++){
    corrEt += m_calibFunc.at(BinID).at(i)*pow(et,(int)i); 
  }
  
  uint16_t jetEtOut = (uint16_t)corrEt;
  
  if(jetEtOut < 1024) {
    return (double)jetEtOut;
  }
  return (double)1023;

}

// Jet Calibration from Frederick(Helsinki), Monika/Creighton (Wisconsin)
double 
corrJetEt1(double et, double eta)
{
  double etabin[23] = {-5.115, -4.439, -3.839, -3.33, 
			-3.0, -2.172, -1.74, -1.392, -1.044, -0.696, -0.348, 
			0.0, 0.348, 0.696, 1.044, 1.392, 1.74, 2.172, 3.0,
			3.33, 3.839, 4.439, 5.115};

  int BinID = 0;
      
  double domainbin_L[22] = {6.52223337753073373e+00,6.64347505748981959e+00,6.78054870174118296e+00,6.75191887554567405e+00,
			    6.60891660595437802e+00,6.57813476381055473e+00,6.96764764481347232e+00,6.77192746888150943e+00,
			    7.16209661824076260e+00,7.09640803784948027e+00,7.29886808171882517e+00,7.29883431473330546e+00,
			    7.24561741344293875e+00,7.05381822724987995e+00,6.52340799679028827e+00,6.96091042775473401e+00,
			    6.69803071767842262e+00,7.79138848427964259e+00,6.78565437835616603e+00,6.71201461174192904e+00,
			    6.60832257380386334e+00,6.54875448717649267e+00};

  double domainbin_U[22] = {8.90225568813317558e+00,1.24483653543590922e+01,1.32037091554958987e+01,1.70036104608977681e+01,
			    3.54325008263432011e+01,4.28758696753095450e+01,4.73079850563588025e+01,4.74182802251108981e+01,
			    4.62509826468679748e+01,5.02198002212212913e+01,4.69817029938948352e+01,4.77263481299233732e+01,
			    4.86083837976362076e+01,4.80105593452927337e+01,5.11550616006504200e+01,4.90703092811585861e+01,
			    4.11879629179572788e+01,3.21820720507165845e+01,1.71844078553560529e+01,1.33158534849654764e+01,
			    1.43586396719878149e+01,1.08629843894704248e+01};

  double A[22] = {2.03682,-4.36824,-4.45258,-6.76524,-22.5984,-24.5289,-24.0313,-22.1896,-21.7818,-22.9882,-20.3321,
		  -21.0595,-22.1007,-22.658,-23.6898,-24.8888,-23.3246,-21.5343,-6.41221,-4.58952,-3.17222,0.637666};

  double B[22] = {0.226303,0.683099,0.704578,0.704623,0.825928,0.80086,0.766475,0.726059,0.760964,0.792227,0.789188,0.795219,
		  0.781097,0.768022,0.740101,0.774782,0.788106,0.814502,0.686877,0.709556,0.628581,0.317453};
  
  double C[22] = {0.00409083,0.000235995,8.22958e-05,2.47567e-05,0.000127995,0.000132914,0.000133342,0.000133035,0.000125993,
		  8.25968e-05,9.94442e-05,9.11652e-05,0.000109351,0.000115883,0.00011112,0.000122559,0.00015868,0.000152601,
		  0.000112927,6.29999e-05,0.000741798,0.00274605};

  double D[22] = {8.24022,7.55916,7.16448,6.31577,5.96339,5.31203,5.35456,4.95243,5.34809,4.93339,5.05723,5.08575,5.18643,5.15202,
		  4.48249,5.2734,5.51785,8.00182,6.21742,6.96692,7.22975,8.12257};
  
  double E[22] = {-0.343598,-0.294067,-0.22529,-0.0718625,0.004164,0.081987,0.124964,0.15006,0.145201,0.182151,0.187525,0.184763,
		  0.170689,0.155268,0.174603,0.133432,0.0719798,-0.0921457,-0.0525274,-0.208415,-0.253542,-0.318476};

  double F[22] = {0.0171799,0.0202499,0.0186897,0.0115477,0.00603883,0.00446235,0.00363449,0.00318894,0.00361997,0.00341508,
		  0.00366392,0.0036545,0.00352303,0.00349116,0.00294891,0.00353187,0.00460384,0.00711028,0.0109351,0.0182924,
		  0.01914,0.0161094};

  for(int i = 0; i < 22; i++){
    if(eta > etabin[i] && eta <= etabin[i+1])
      BinID = i;
  }

  if(et >= domainbin_U[BinID]){
    return 2*(et-A[BinID])/(B[BinID]+sqrt(B[BinID]*B[BinID]-4*A[BinID]*C[BinID]+4*et*C[BinID]));
  }
  else if(et > domainbin_L[BinID]){
    return 2*(et-D[BinID])/(E[BinID]+sqrt(E[BinID]*E[BinID]-4*D[BinID]*F[BinID]+4*et*F[BinID]));
  }
  else return et;
}


// EM correction from ORCA for cmsim 133
double 
//corrEmEt(double et, double eta) {
corrEmEt(double et, int eta) {


  const int nscales = 32;
  /*
  const int nscales = 27;
  double etalimit[nscales] = { 0.0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.870,0.957,
			  1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,
			  2.043,2.172,2.322,2.500};
  */

  double scfactor[nscales] = { 
    1.00,1.01,1.02,1.02,1.02,1.06,1.04,1.04,1.05,1.09,1.10,1.10,1.15,
    1.20,1.27,1.29,1.32,1.52,1.52,1.48,1.40,1.32,1.26,1.21,1.17,1.15, 
    1.15,1.15,1.15,1.15,1.15,1.15};

  /*
  double scale=1.;
  for (int i=0;i<nscales;i++) {
    if (std::abs(eta)<etalimit[i]) {
      scale = scfactor[i];
    }
  }
    return (scale*et);
  */

  if (eta>=0 && eta <=28)
    return (scfactor[eta]*et);
  else
    return et;
}


// Rounding the Et info for simulating the regional Et resolution
double 
RCTEnergyTrunc(double et, double LSB, double thres) {

  //return et;
  if (et>=thres) return thres;

  //et += LSB/2.;
  //double ret = (int)(et / LSB) * LSB + LSB;
  int iEt = (int)(et / LSB);
  double ret =  (double)iEt * LSB;

  return ret;
}


double 
GCTEnergyTrunc(double et, double LSB, bool doEM) {

  //return et;

  //double L1CaloEmEtScaleLSB = LSB;
  //double L1CaloRegionEtScaleLSB = LSB;

  //if (et>0.) et += LSB/2.; // round up

  double L1CaloEmThresholds[64] = { 
    0.,     1.,     2.,     3.,     4.,     5.,     6.,     7.,     8.,     9., 
    10.,    11.,    12.,    13.,    14.,    15.,    16.,    17.,    18.,    19., 
    20.,    21.,    22.,    23.,    24.,    25.,    26.,    27.,    28.,    29., 
    30.,    31.,    32.,    33.,    34.,    35.,    36.,    37.,    38.,    39., 
    40.,    41.,    42.,    43.,    44.,    45.,    46.,    47.,    48.,    49.,
    50.,    51.,    52.,    53.,    54.,    55.,    56.,    57.,    58.,    59., 
    60.,    61.,    62.,    63.
  };
 
  double L1CaloJetThresholds[64] = {       
    0.,     10.,    12.,    14.,    15.,    18.,    20.,    22.,    24.,    25.,
    28.,    30.,    32.,    35.,    37.,    40.,    45.,    50.,    55.,    60.,    
    65.,    70.,    75.,    80.,    85.,    90.,    100.,   110.,   120.,   125.,   
    130.,   140.,   150.,   160.,   170.,   175.,   180.,   190.,   200.,   215.,   
    225.,   235.,   250.,   275.,   300.,   325.,   350.,   375.,   400.,   425.,   
    450.,   475.,   500.,   525.,   550.,   575.,   600.,   625.,   650.,   675.,   
    700.,   725.,   750.,   775.
  };
  


  double L1CaloThresholds[64];
  if (doEM) {
    for (int i=0;i<64;i++)
      L1CaloThresholds[i] = L1CaloEmThresholds[i];
  } else {
    for (int i=0;i<64;i++)
      L1CaloThresholds[i] = L1CaloJetThresholds[i];
  }


  double ret = 0.;
  for (int i=63;i>0;i--) {
    if (et>=(L1CaloThresholds[i])) {
      if (i==63) {
	ret = L1CaloThresholds[63];
      } else {
	/*
	double minL = std::abs(et - L1CaloThresholds[i]); 
	double minU = std::abs(et - L1CaloThresholds[i+1]); 
	if (minL<minU)
	  ret = L1CaloThresholds[i];
	else
	  ret = L1CaloThresholds[i+1];
	*/
	/*
	if (doEM) {
	  ret = L1CaloThresholds[i];
	} else {
	  ret = L1CaloThresholds[i+1];
	}
	*/
	ret = L1CaloThresholds[i];
      }
      break;
    }
  }
  return ret;
}



