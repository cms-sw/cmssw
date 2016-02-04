#include "DQM/HcalMonitorTasks/interface/HcalCaloTowerMonitor.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

HcalCaloTowerMonitor::HcalCaloTowerMonitor(){}

HcalCaloTowerMonitor::~HcalCaloTowerMonitor() {}

void HcalCaloTowerMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"CaloTowerMonitor";
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  std::cout << "CaloTower Monitor eta min/max set to " << etaMin_ << "/" << etaMax_ << std::endl;

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  std::cout << "CaloTower Monitor phi min/max set to " << phiMin_ << "/" << phiMax_ << std::endl;

  ievt_=0;
  // book histograms
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_); 
      meEVT_ = m_dbe->bookInt("CaloTower Event Number");     
      meEVT_->Fill(ievt_); 

      // Calo Tower occupancy
      caloTowerOccMap=m_dbe->book2D("CaloTowerOccupancyMap", "Calo Tower Occupancy Map",
				    etaBins_,etaMin_,etaMax_,
				    phiBins_,phiMin_,phiMax_);
      // Calo Tower energy
      caloTowerEnergyMap=m_dbe->book2D("CaloTowerEnergyMap", "Calo Tower Energy Map",
				       etaBins_,etaMin_,etaMax_,
				       phiBins_,phiMin_,phiMax_);

      caloTowerTime = m_dbe->book1D("CaloTowerTime", "Calo Tower Time",
				    10,0,10);
      caloTowerEnergy = m_dbe->book1D("CalotowerEnergy", "Calo Tower Energy",
				      100,0,100);
      caloTowerMeanEnergyEta = m_dbe->bookProfile("CaloTowerMeanEnergyEta","Calo tower Mean Energy vs. Eta",
						  etaBins_,etaMin_,etaMax_,100,0,100);

      // Make plots of HCAL contributions to tower
      m_dbe->setCurrentFolder(baseFolder_+"/"+"HCAL");
      hcalOccMap=m_dbe->book2D("HcalOccupancyMap", "Calo Tower Occupancy Map",
			       etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
      // Calo Tower energy
      hcalEnergyMap=m_dbe->book2D("HcalEnergyMap", "Calo Tower Energy Map",
				  etaBins_,etaMin_,etaMax_,
				  phiBins_,phiMin_,phiMax_);

      hcalTime = m_dbe->book1D("HcalTime", "Calo Tower Time",
				    10,0,10);
      hcalEnergy = m_dbe->book1D("HcalEnergy", "Calo Tower Energy",
				      100,0,100);
      hcalMeanEnergyEta = m_dbe->bookProfile("HcalMeanEnergyEta",
					     "Calo tower Mean Energy vs. Eta",
					     etaBins_,etaMin_,etaMax_,
					     100,0,100);
      

      m_dbe->setCurrentFolder(baseFolder_+"/"+"ECAL");
      ecalOccMap=m_dbe->book2D("EcalOccupancyMap", "Calo Tower Occupancy Map",
			       etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
      // Calo Tower energy
      ecalEnergyMap=m_dbe->book2D("EcalEnergyMap", "Calo Tower Energy Map",
				  etaBins_,etaMin_,etaMax_,
				  phiBins_,phiMin_,phiMax_);

      ecalTime = m_dbe->book1D("EcalTime", "Calo Tower Time",
			       10,0,10);
      ecalEnergy = m_dbe->book1D("EcalEnergy", "Calo Tower Energy",
				 100,0,100);
      ecalMeanEnergyEta = m_dbe->bookProfile("EcalMeanEnergyEta",
					     "Calo tower Mean Energy vs. Eta",
					     etaBins_,etaMin_,etaMax_,
					     100,0,100);

      // Comparison Plots
      m_dbe->setCurrentFolder(baseFolder_+"/"+"ComparisonPlots");
      time_HcalvsEcal=m_dbe->book2D("HcalvsEcalTime",
				    "Hcal Time vs. Ecal time",
				    10,0,10,10,0,10);
      time_CaloTowervsEcal=m_dbe->book2D("CaloTowervsEcalTime",
					 "Calotower Time vs. Ecal time",
					 10,0,10,10,0,10);
      time_CaloTowervsHcal=m_dbe->book2D("CaloTowervsHcalTime",
					 "CaloTower Time vs. Hcal time",
					 10,0,10,10,0,10);
      energy_HcalvsEcal=m_dbe->book2D("HcalvsEcalEnergy",
				    "Hcal Energy vs. Ecal energy",
				    100,0,100,100,0,100);
      energy_CaloTowervsEcal=m_dbe->book2D("CaloTowervsEcalEnergy",
					 "Calotower Energy vs. Ecal energy",
					 100,0,100,100,0,100);
      energy_CaloTowervsHcal=m_dbe->book2D("CaloTowervsHcalEnergy",
					 "CaloTower Energy vs. Hcal energy",
					 100,0,100,100,0,100);
    } // if (m_dbe)
  
} //void HcalCaloTowerMonitor::setup(...)


void HcalCaloTowerMonitor::processEvent(const CaloTowerCollection& calotower)
{
  // fill histograms for each event
  if(!m_dbe)  
    { 
      if(fVerbosity) std::cout <<"<HcalCaloTowerMonitor::processEvent>    DQMStore not instantiated!!!\n"; 
      return; 
    }  // if(!m_dbe)
  
  if (fVerbosity) std::cout <<"Processing calotower"<<std::endl;


  // Store values of hcal, energy for forming averages at each eta
  float hcalenergy[90]={0.};
  float ecalenergy[90]={0.};
  int etacounts[90]={0};

  for (CaloTowerCollection::const_iterator it=calotower.begin();
       it!=calotower.end();++it)
    {
      //caloTowers give eta and phi; need to map to ieta,iphi
      // phi is mapped into 72 equally spaced segments (2pi/72=.087 radians each), starting at iphi=1
      // at ieta=21 (eta=1.74), iphi spaced in 10 degree increments 
      // at ieta=40 (eta=4.716), iphi spaced in 20 degree increments
      // However, segmentation values are similar
      // (iphi=1,2,3,4,5 for 5 degree segments,
      //  iphi 1,3,5,... for 10 degree segments, and iphi=3,7,... for 20 deg.)
      int iphi;
      double eta=it->eta();
      double phi=it->phi();
      // if phi<0; shift it by 2pi (so that phi runs from 0->2pi, rather than -pi->pi)
      if (phi<0)
	phi+=2*3.14159265359;
      iphi = int(phi/.087)+1;
      if (fabs(eta)>1.74)
	{
	  if (fabs(eta)<=4.716)
	    iphi=(iphi-1)/2*2+1;
	  else
	    iphi=(iphi-1)/4*4+3;
	  
	}
      //ieta is more complicated -- it runs in segments of .087 for the first 20 segments, then the segmentation becomes non-uniform
      int ieta=getIeta(eta);

      // Add 42 to the indices.  Only index[1...] will have values within [HB...HF].  Index[0] and [84] store values outside the expected calorimeter range.
      hcalenergy[ieta+42]+=it->hadEnergy();
      ecalenergy[ieta+42]+=it->emEnergy();
      etacounts[ieta+42]++;

      /*
      std::cout <<"CALOTOWER eta = \t"<<eta<<"\tphi = "<<it->phi()<<std::endl;
      std::cout <<"\t ieta = \t"<<ieta<<"\tiphi = "<<iphi<<std::endl;
      */

      // Fill histograms

      // calotowers
      caloTowerOccMap->Fill(ieta,iphi);
      caloTowerEnergyMap->Fill(ieta,iphi,it->energy());
      caloTowerTime->Fill((it->ecalTime()+it->hcalTime())/2.);
      caloTowerEnergy->Fill(it->energy());
      //std::cout <<"calotower times: "<<it->ecalTime()<<"  "<<it->hcalTime()<<std::endlx;

      //hcal
      hcalOccMap->Fill(ieta,iphi);
      hcalEnergyMap->Fill(ieta,iphi,it->hadEnergy());
      hcalTime->Fill(it->hcalTime());
      hcalEnergy->Fill(it->hadEnergy());

      //ecal
      ecalOccMap->Fill(ieta,iphi);
      ecalEnergyMap->Fill(ieta,iphi,it->emEnergy());
      ecalTime->Fill(it->ecalTime());
      ecalEnergy->Fill(it->emEnergy());

      time_HcalvsEcal->Fill(it->ecalTime(),it->hcalTime());
      energy_HcalvsEcal->Fill(it->emEnergy(),it->hadEnergy());
    } // for CaloTowerCollection::const_iterator it...

  // Now form average for each eta bin;
  for (int i=0;i<90;++i)
    {
      if (etacounts[i]==0) continue; 
      caloTowerMeanEnergyEta->Fill(i-42, 1.*(hcalenergy[i]+ecalenergy[i])/etacounts[i]);
      hcalMeanEnergyEta->Fill(i-42, 1.*hcalenergy[i]/etacounts[i]);
      ecalMeanEnergyEta->Fill(i-42,1.*ecalenergy[i]/etacounts[i]);
    }
}

int HcalCaloTowerMonitor::getIeta(double eta)
{
  // return Ieta index given physical eta value

  double myeta=fabs(eta);
  int neg=int(eta/myeta); // eta can be negative or positive

  if (myeta<=1.740) // first 20 bins are spaced evenly in increments of .087 radians, starting at ieta=1
    return (1+int(myeta/.087))*neg;

  if (fabs(eta)<=1.83)
    return int(21*neg);
  if (fabs(eta)<=1.93)
    return int(22*neg);
  if (fabs(eta)<=2.043)
    return int(23*neg);
  if (fabs(eta)<=2.172)
    return int(24*neg);
  if (fabs(eta)<=2.322)
    return int(25*neg);
	     
  if (fabs(eta)<=2.5)
    return int(26*neg);
  if (fabs(eta)<=2.65)
    return int(27*neg);

  // 28 and 29 aren't quite right -- eta values depend on depth
  if (fabs(eta)<=2.853)
    return int(28*neg);
  if (fabs(eta)<=2.964)
    return int(29*neg);
  if (fabs(eta)<=3.139)
    return int(30*neg);
		 
  if (fabs(eta)<=3.314)
    return int(31*neg);
  if (fabs(eta)<=3.489)
    return int(32*neg);
  if (fabs(eta)<=3.664)
    return int(33*neg);
  if (fabs(eta)<=3.839)
    return int(34*neg);
  if (fabs(eta)<=4.013)
    return int(35*neg);
	 
  if (fabs(eta)<=4.191)
    return int(36*neg);
  if (fabs(eta)<=4.363)
    return int(37*neg);
  if (fabs(eta)<=4.538)
    return int(38*neg);
  if (fabs(eta)<=4.716)
    return int(39*neg);
  if (fabs(eta)<=4.889)
    return int(40*neg);
  if (fabs(eta)<=5.191)
    return int(41*neg);

  // Anything with large eta is outside calorimeter; skip it?
  // Are there ZDC-based calotowers?
  return 42;
    
} // int HcalCaloTowerMonitor::getIeta(double eta)


void HcalCaloTowerMonitor::reset(){}
