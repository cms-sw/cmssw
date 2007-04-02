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
// $Id: FastL1Region.cc,v 1.5 2007/03/02 23:00:41 chinhan Exp $
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

  bool printIt = false;
  double sEta = 2.1;
  double sPhi = -3.0;
  double dEta = 0.55;
  double dPhi = 0.55;
  int sRun = 1;
  int sEvent = 51;

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
	
	/*
	std::cout<<"********************************************"<<std::endl;
	std::cout<<detId.ieta()<<" | "<<detId.iphi()<<std::endl;
	std::cout<<Towers[i].id().ieta()<<" | "<<Towers[i].id().iphi()<<std::endl;
	std::cout<<crIeta<<" | "<<crIphi<<std::endl;
	std::cout<<"Tower eta,phi:" <<Towers[i].eta()<<" | "<<Towers[i].phi()<<std::endl;
	std::cout<<"Crys. eta,phi:" <<gP.eta()<<" | "<<gP.phi()<<std::endl;
	std::cout<<i<<" | "<<crIeta + 5*crIphi<<std::endl;
	*/

	EMCrystalEnergy[i][crIeta + 5*crIphi] = recHit.energy();
      }
    }  
  }
  
  // After having filled crsystal info set all veto bits
  SetTowerBits();
 
  if (GetiEta()==4 || GetiEta()==5 ||  GetiEta()==6 ||  
      GetiEta()==15 || GetiEta()==16 || GetiEta()==17) {
    // EE FG bits are filled here!!!
    //e.getByLabel(la[1].first,la[1].second,ec);
    e.getByLabel(Config.EmInputs.at(1),ec);
    ethres = Config.CrystalEEThreshold;
    double towerEnergy[16];
    // loop over towers
    for(int i=0;i<16;i++) {
      fgBit[i] = false; // re-iniate
      if ((int)e.id().event() == sEvent && (int)e.id().run() == sRun
	  && Towers[i].eta() > (sEta-dEta) && Towers[i].eta() < (sEta+dEta) && 
	  Towers[i].phi() > (sPhi-dPhi) && Towers[i].phi() < (sPhi+dPhi)
	  && printIt ) 
	{
	  std::cout<<"-----------------------------------------------"<<std::endl;
	  std::cout<<"Run: " << e.id().run() << " Event: "<< e.id().event()<< std::endl;  
	  std::cout<<"Tower (et,eta,phi): "<<Towers[i].et()<<", "<<Towers[i].eta()<<", "<<
	    Towers[i].phi()<<std::endl;
	  std::cout<<" Tower (ieta,iphi): "<<Towers[i].id().ieta()<<", "
		   <<Towers[i].id().ieta()<<std::endl;
	}
            
      //if (Towers[i].hadEt()>Config.HadNoiseLevel && Towers[i].emEt()>Config.EMNoiseLevel ) {
      if (Towers[i].emEt()>Config.EMNoiseLevel ) {
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
	    if (maxRecHit2.energy()<maxRecHit.energy()) {
	      maxRecHit2 = maxRecHit;
	    }	
	    maxRecHit = recHit;
	  }	
	}
      }  
      
      if ((int)e.id().event() == sEvent && (int)e.id().run() == sRun
	  && Towers[i].eta() > (sEta-dEta) && Towers[i].eta() < (sEta+dEta) && 
	  Towers[i].phi() > (sPhi-dPhi) && Towers[i].phi() < (sPhi+dPhi)
	  && printIt ) 
	{
	  std::cout<<"MaxHit found E: " << maxRecHit.energy()<< std::endl;  
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
	  //std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
	  if ((int)e.id().event() == sEvent && (int)e.id().run() == sRun
	      && Towers[i].eta() > (sEta-dEta) && Towers[i].eta() < (sEta+dEta) && 
	      Towers[i].phi() > (sPhi-dPhi) && Towers[i].phi() < (sPhi+dPhi)
	      && printIt ) 
	    {
	      std::cout<<"Crystal in same tower found!!!"<< std::endl;  
	    }
	  
	  if ( (!westV.empty() && recHit.detid()==westV[0]) || 
	       (!eastV.empty() && recHit.detid()==eastV[0]) || 
	       (!northV.empty() && recHit.detid()==northV[0]) || 
	       (!southV.empty() && recHit.detid()==southV[0]) 
	       ) {
	    //if (maxRecHit2.energy()<recHit.energy()) {
	    //  maxRecHit2 = recHit;
	    //}	
	    max2En += recHit.energy();
	    if ((int)e.id().event() == sEvent && (int)e.id().run() == sRun
		&& Towers[i].eta() > (sEta-dEta) && Towers[i].eta() < (sEta+dEta) && 
		Towers[i].phi() > (sPhi-dPhi) && Towers[i].phi() < (sPhi+dPhi)
		&& printIt ) 
	      {
		std::cout<<"Neighbour Crystal enrgy added: "<<recHit.energy()<< std::endl;  
	      }
	    
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
      
      if ((int)e.id().event() == sEvent && (int)e.id().run() == sRun
	  && Towers[i].eta() > (sEta-dEta) && Towers[i].eta() < (sEta+dEta) && 
	  Towers[i].phi() > (sPhi-dPhi) && Towers[i].phi() < (sPhi+dPhi)
	  && printIt ) 
	{
	  std::cout<<"********************************************"<<std::endl;
	  std::cout<<"TotE,TwrE: "<<totE<<" | "<<towerEnergy[i]<<std::endl;	
	  std::cout<<"Ratio (cut 0.4): "<<totE/towerEnergy[i]<< std::endl;  
	  std::cout<<"Rgn ieta/iphi: "<<GetiEta()<<" | "<<i<<": "<<fgBit[i]<<std::endl;
	}   
      
    }
  }
   
 
}

//
void
FastL1Region::FillTower(const CaloTower& t, int& tid) 
{
  Towers[tid] = CaloTower(t);
}



//
void
FastL1Region::FillTower_Scaled(const CaloTower& t, int& tid) 
{
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

  Towers[tid] = CaloTower(t.id(),t.momentum(),emScale * t.emEt(),hadScale * t.hadEt(),t.outerEt(),0,0);
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

  bool printIt = false;
  double sEta = 2.1;
  double sPhi = -3.0;
  double dEta = 0.55;
  double dPhi = 0.55;
  int sRun = 1;
  int sEvent = 51;

  //double cenEta = Towers[6].eta();
  //double cenPhi = Towers[6].phi();
  double cenEta = sEta;
  double cenPhi = sPhi;

  float emThres = Config.EMActiveLevel;
  float hadThres = Config.HadActiveLevel;

  // init pattern containers
  unsigned emEtaPat = 0;
  unsigned emPhiPat = 0;
  unsigned hadEtaPat = 0;
  unsigned hadPhiPat = 0;
  unsigned one = 1;

  if ((int)iEvent.id().event() == sEvent && (int)iEvent.id().run() == sRun
      && cenEta > (sEta-dEta) && cenEta < (sEta+dEta) && cenPhi > (sPhi-dPhi) && cenPhi < (sPhi+dPhi)
      && printIt ) {
    std::cout<<"Region Pattern -----------------------------------------------"<<std::endl;
    std::cout<<"Region (ieta,iphi): "<<GetiEta()<<", "<<GetiPhi()<<std::endl;
    std::cout<<"Tower center (eta,phi): "<<cenEta<<", "<<cenPhi<<std::endl;
    for (int i=0; i<16; i++) {
      std::cout<<Towers[i].emEt()<<" ";
      if ((i%4)==3) std::cout<<std::endl;
    }
    std::cout<<"Em Pattern -----"<<std::endl;
    for (int i=0; i<16; i++) {
      if (Towers[i].emEt() > emThres)
	std::cout<<"1 ";
      else
	std::cout<<"0 ";
      if ((i%4)==3) std::cout<<std::endl;
    }
    std::cout<<"Had Pattern ***"<<std::endl;
    for (int i=0; i<16; i++) {
      std::cout<<Towers[i].hadEt()<<" ";
      if ((i%4)==3) std::cout<<std::endl;
    }
    std::cout<<"-----"<<std::endl;
    for (int i=0; i<16; i++) {
      if( Towers[i].hadEt() > hadThres)
	std::cout<<"1 ";
      else
	std::cout<<"0 ";
      if ((i%4)==3) std::cout<<std::endl;
    }
  }


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

  // equivalent code:
  /*
  for (int i=0; i<16; i++) {
    if(Towers[i].emEt() > emThres) {
      //emEtaPat |= ((unsigned)pow(2,i%4));
      //emPhiPat |= ((unsigned)pow(2,i/4));
      emEtaPat |= ((unsigned)pow(2,i%4));
      emPhiPat |= ((unsigned)pow(2,i/4));
    }

    if( Towers[i].hadEt() > hadThres) {
      hadEtaPat |= ((unsigned)pow(2,i%4));
      hadPhiPat |= ((unsigned)pow(2,i/4));
    }

  }
  */

  if ((int)iEvent.id().event() == sEvent && (int)iEvent.id().run() == sRun
      && cenEta > (sEta-dEta) && cenEta < (sEta+dEta) && cenPhi > (sPhi-dPhi) && cenPhi < (sPhi+dPhi)
      && printIt ) {
    unsigned etaPattern = emEtaPat | hadEtaPat;
    unsigned phiPattern = emPhiPat | hadPhiPat;

    std::cout<<"em eta pattern: "<<emEtaPat<<std::endl;  
    std::cout<<"em phi pattern: "<<emPhiPat<<std::endl;  
    std::cout<<"hd eta pattern: "<<hadEtaPat<<std::endl;  
    std::cout<<"hd phi pattern: "<<hadPhiPat<<std::endl;  
    std::cout<<"eta pattern: "<<etaPattern<<std::endl;  
    std::cout<<"phi pattern: "<<phiPattern<<std::endl;  
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

  
  //std::cout<<"*******************************************"<<std::endl;
  //std::cout<<"Tower Pattern: "<<std::endl;
  //std::cout<<emEtaPat<<std::endl;
  //std::cout<<emPhiPat<<std::endl;
  //std::cout<<hadEtaPat<<std::endl;
  //std::cout<<hadPhiPat<<std::endl;
  //std::cout<<emEtaPat<<" | "<<emEtaPattern<<std::endl;
  //std::cout<<emPhiPat<<" | "<<emPhiPattern<<std::endl;
  //std::cout<<hadEtaPat<<" | "<<hadEtaPattern<<std::endl;
  //std::cout<<hadPhiPat<<" | "<<hadPhiPattern<<std::endl;


  for(std::vector<unsigned>::iterator i = vetoPatterns.begin();
      i != vetoPatterns.end();  i++) {
    unsigned etaPattern = emEtaPat | hadEtaPat;
    unsigned phiPattern = emPhiPat | hadPhiPat;
    if(etaPattern == *i || phiPattern == *i) // combined pattern
      //if(emEtaPat == *i || emPhiPat == *i || hadEtaPat == *i || hadPhiPat == *i)
      {
	tauBit = true;

	/*
	std::cout<<"************************ tauveto fired!!! *****************************************"<<std::endl;
	std::cout << "Run: " << iEvent.id().run() << " Event: " << iEvent.id().event()<< std::endl;  
	if(emEtaPat == *i || emPhiPat == *i || hadEtaPat == *i || hadPhiPat == *i) {
	  std::cout<<"******* alternate tauveto fired as well! *******"<<std::endl;
	} else {
	  std::cout<<"####### alternate tauveto didn't fire! #######"<<std::endl;
	}

	std::cout<<"Region Pattern -----------------------------------------------"<<std::endl;
	std::cout<<"Region (ieta,iphi): "<<GetiEta()<<", "<<GetiPhi()<<std::endl;
	std::cout<<"Tower center (eta,phi): "<<cenEta<<", "<<cenPhi<<std::endl;
	for (int i=0; i<16; i++) {
	  std::cout<<Towers[i].emEt()<<" ";
	  if ((i%4)==3) std::cout<<std::endl;
	}
	std::cout<<"Em Pattern -----"<<std::endl;
	for (int i=0; i<16; i++) {
	  if (Towers[i].emEt() > emThres)
	    std::cout<<"1 ";
	  else
	    std::cout<<"0 ";
	  if ((i%4)==3) std::cout<<std::endl;
	}
	std::cout<<"Had Pattern ***"<<std::endl;
	for (int i=0; i<16; i++) {
	  std::cout<<Towers[i].hadEt()<<" ";
	  if ((i%4)==3) std::cout<<std::endl;
	}
	std::cout<<"-----"<<std::endl;
	for (int i=0; i<16; i++) {
	  if( Towers[i].hadEt() > hadThres)
	    std::cout<<"1 ";
	  else
	    std::cout<<"0 ";
	  if ((i%4)==3) std::cout<<std::endl;
	}
	
	std::cout<<"em eta pattern: "<<emEtaPat<<std::endl;  
	std::cout<<"em phi pattern: "<<emPhiPat<<std::endl;  
	std::cout<<"hd eta pattern: "<<hadEtaPat<<std::endl;  
	std::cout<<"hd phi pattern: "<<hadPhiPat<<std::endl;  
	std::cout<<"eta pattern: "<<etaPattern<<std::endl;  
	std::cout<<"phi pattern: "<<phiPattern<<std::endl;  
	*/
	
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

    double EThres = 0.;
    double HThres = 0.;
    double EBthres = Config.TowerEBThreshold;
    double HBthres = Config.TowerHBThreshold;
    double EEthres = Config.TowerEBThreshold;
    double HEthres = Config.TowerEEThreshold;
    //if(std::abs(Towers[i].eta())<1.479) {
    if(std::abs(Towers[i].eta())<2.322) {
      EThres = EBthres;
    } else {
      EThres = EEthres;
    }
    //if(std::abs(Towers[i].eta())<1.305) {
    if(std::abs(Towers[i].eta())<2.322) {
      HThres = HBthres;
    } else {
      HThres = HEthres;
    }

    if ( Towers[i].emEt() >= EThres )
      sumet += Towers[i].emEt();
    if ( Towers[i].hadEt() >= HThres )
      sumet += Towers[i].hadEt();

  }
  return sumet;
}

double 
FastL1Region::CalcSumE()
{
  double sume=0;
  for (int i=0; i<16; i++) {
    double EThres = 0.;
    double HThres = 0.;
    double EBthres = Config.TowerEBThreshold;
    double HBthres = Config.TowerHBThreshold;
    double EEthres = Config.TowerEBThreshold;
    double HEthres = Config.TowerEEThreshold;
    //if(std::abs(Towers[i].eta())<1.479) {
    if(std::abs(Towers[i].eta())<2.322) {
      EThres = EBthres;
    } else {
      EThres = EEthres;
    }
    //if(std::abs(Towers[i].eta())<1.305) {
    if(std::abs(Towers[i].eta())<2.322) {
      HThres = HBthres;
    } else {
      HThres = HEthres;
    }

    if ( Towers[i].emEt() >= EThres )
      sume += Towers[i].emEnergy();
    if ( Towers[i].hadEt() >= HThres )
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

