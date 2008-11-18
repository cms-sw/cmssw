#include "DQM/HcalMonitorTasks/interface/HcalCaloTowerMonitor.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

HcalCaloTowerMonitor::HcalCaloTowerMonitor(){}

HcalCaloTowerMonitor::~HcalCaloTowerMonitor() {}

void HcalCaloTowerMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"CaloTowerMonitor";
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "CaloTower Monitor eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "CaloTower Monitor phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  ievt_=0;
  // book histograms
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_); 
      meEVT_ = m_dbe->bookInt("CaloTower Event Number");     
      meEVT_->Fill(ievt_); 

      // Calo Tower occupancy
      caloTowerOcc=m_dbe->book2D("CaloTowerOccupancy", "Calo Tower Occupancy",
				 etaBins_,etaMin_,etaMax_,
				 phiBins_,phiMin_,phiMax_);
      // Calo Tower energy
     caloTowerEnergy=m_dbe->book2D("CaloTowerEnergy", "Calo Tower Energy",
				 etaBins_,etaMin_,etaMax_,
				 phiBins_,phiMin_,phiMax_);
    } // if (m_dbe)

} //void HcalCaloTowerMonitor::setup(...)


void HcalCaloTowerMonitor::processEvent(const CaloTowerCollection& calotower)
{
  // fill histograms for each event
  if(!m_dbe)  
    { 
      if(fVerbosity) cout <<"<HcalCaloTowerMonitor::processEvent>    DQMStore not instantiated!!!\n"; 
      return; 
    }  // if(!m_dbe)
  

  for (CaloTowerCollection::const_iterator it=calotower.begin();
       it!=calotower.end();++it)
    {
      //caloTowers give eta and phi; need to map to ieta,iphi
      // phi is mapped into 72 equally spaced segments (2pi/72=.087 radians each), starting at iphi=1
      // at ieta=21 (eta=1.74), iphi spaced in 10 degree increments 
      // at ieta=40 (eta=4.716), iphi spaced in 20 degree increments
      // However, segmentation values are similar
      // (iphi=1,2,3,4,5 for 5 degree segments,
      //  iphi 1,3,5,... for 10 degree segments, and iphi=1,5,... for 20 deg.)
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
	    iphi=(iphi-1)/4*4+1;
	  
	}
      //ieta is more complicated -- it runs in segments of .087 for the first 20 segments, then the segmentation becomes non-uniform
      int ieta=getIeta(eta);
      /*
      cout <<"CALOTOWER eta = \t"<<eta<<"\tphi = "<<it->phi()<<endl;
      cout <<"\t ieta = \t"<<ieta<<"\tiphi = "<<iphi<<endl;
      */
      // Fill histograms
      caloTowerOcc->Fill(ieta,iphi);
      caloTowerEnergy->Fill(ieta,iphi,it->energy());
    } // for CaloTowerCollection::const_iterator it...

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

  // Anything with large eta is outside calorimeter; skip it
  return 42;
    
} // int HcalCaloTowerMonitor::getIeta(double eta)


void HcalCaloTowerMonitor::reset(){}
