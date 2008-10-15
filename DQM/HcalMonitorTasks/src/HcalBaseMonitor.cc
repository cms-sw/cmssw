#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#define ETAMAX 44.5
#define ETAMIN -44.5
#define PHIMAX 73.5
#define PHIMIN -0.5

HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
  hotCells_.clear();
  rootFolder_ = "Hcal";
  baseFolder_ = "BaseMonitor";
}

HcalBaseMonitor::~HcalBaseMonitor() {}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  hotCells_ =  ps.getUntrackedParameter<vector<string> >( "HotCells" );
  
  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  rootFolder_ = subsystemname + "/";

  // Global cfgs
  
  fVerbosity = ps.getUntrackedParameter<int>("debug",0); // shouldn't fVerbosity be an int32?
  makeDiagnostics=ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);
  showTiming = ps.getUntrackedParameter<bool>("showTiming",false);
  
  checkHB_ = ps.getUntrackedParameter<bool>("checkHB",true);
  checkHE_ = ps.getUntrackedParameter<bool>("checkHE",true);
  checkHO_ = ps.getUntrackedParameter<bool>("checkHO",true);
  checkHF_ = ps.getUntrackedParameter<bool>("checkHF",true);



  // Minimum error rate that will caused the problem histogram to be filled
  minErrorFlag_ = ps.getUntrackedParameter<double>("minErrorFlag",0.05);
  
  // Set eta, phi boundaries
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 44.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", ETAMIN);
    
  if (etaMax_ > 44.5)
    {
      cout <<"<HcalPedestalMonitor> WARNING:  etaMax_ value of "<<etaMax_<<" exceeds maximum allowed value of 44.5"<<endl;
      cout <<"                      Value being set back to 44.5."<<endl;
      cout <<"                      Additional code changes are necessary to allow value of "<<etaMax_<<endl;
      etaMax_ = 44.5;
    }

  if (etaMin_ < ETAMIN)
    {
      cout <<"<HcalPedestalMonitor> WARNING:  etaMin_ value of "<<etaMin_<<" exceeds minimum allowed value of 44.5"<<endl;
      cout <<"                      Value being set back to ETAMIN."<<endl;
      cout <<"                      Additional code changes are necessary to allow value of "<<etaMin_<<endl;
      etaMin_ = ETAMIN;
    }

  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", PHIMAX);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", PHIMIN);
  phiBins_ = (int)(phiMax_ - phiMin_);


  return;
}

void HcalBaseMonitor::done(){}

void HcalBaseMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();    
    
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    m_dbe->removeContents();
    
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    m_dbe->removeContents();
  }
  return;
} // void HcalBaseMonitor::clearME();

// ***************************************************************** //


bool HcalBaseMonitor::vetoCell(HcalDetId id)
{
  /*
    Function identifies whether cell width HcalDetId 'id' should be vetoed, based on elements stored in  hotCells_ array.
  */

  if(hotCells_.size()==0) return false;
  for(unsigned int i = 0; i< hotCells_.size(); ++i)
    {
      
      unsigned int badc = atoi(hotCells_[i].c_str());
      if(id.rawId() == badc) return true;
    }
  return false;
} // bool HcalBaseMonitor::vetoCell(HcalDetId id)





// ************************************************************************************************************ //

bool HcalBaseMonitor::validDetId(HcalSubdetector sd, int ies, int ip, int dp)
{
  // inputs are (subdetector, ieta, iphi, depth)
  // stolen from latest version of DataFormats/HcalDetId/src/HcalDetId.cc (not yet available in CMSSW_2_1_9)

  const int ie ( abs( ies ) ) ;

  return ( ( ip >=  1         ) &&
	   ( ip <= 72         ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 14         ) &&
		   ( dp ==  1         )    ) ||
		 ( ( ( ie == 15 ) || ( ie == 16 ) ) && 
		   ( dp <= 2          )                ) ) ) ||
	     (  ( sd == HcalEndcap ) &&
		( ( ( ie == 16 ) &&
		    ( dp ==  3 )          ) ||
		  ( ( ie == 17 ) &&
		    ( dp ==  1 )          ) ||
		  ( ( ie >= 18 ) &&
		    ( ie <= 20 ) &&
		    ( dp <=  2 )          ) ||
		  ( ( ie >= 21 ) &&
		    ( ie <= 26 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie >= 27 ) &&
		    ( ie <= 28 ) &&
		    ( dp <=  3 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie == 29 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         )          )      ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= 29 ) &&
		    ( ie <= 39 ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= 40 ) &&
		    ( ie <= 41 ) &&
		    ( ip%4 == 3 )         )  ) ) ) ) ;



} // bool  HcalPedestalMonitor::validDetId(HcalSubdetector sd, int ies, int ip, int dp)
