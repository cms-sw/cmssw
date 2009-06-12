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
  
  fVerbosity      = ps.getUntrackedParameter<int>("debug",0); 
  makeDiagnostics = ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);
  fillUnphysical_ = ps.getUntrackedParameter<bool>("fillUnphysicalIphi", true);
  showTiming      = ps.getUntrackedParameter<bool>("showTiming",false);
  dump2database   = ps.getUntrackedParameter<bool>("dump2database",false); // dumps output to database file 
  checkHB_ = ps.getUntrackedParameter<bool>("checkHB",true);
  checkHE_ = ps.getUntrackedParameter<bool>("checkHE",true);
  checkHO_ = ps.getUntrackedParameter<bool>("checkHO",true);
  checkHF_ = ps.getUntrackedParameter<bool>("checkHF",true);
  checkZDC_ = ps.getUntrackedParameter<bool>("checkZDC",true);
  checkNevents_ = ps.getUntrackedParameter<int>("checkNevents",100);


  // Minimum error rate that will caused the problem histogram to be filled
  minErrorFlag_ = ps.getUntrackedParameter<double>("minErrorFlag",0.05);
  
  // Set eta, phi boundaries
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", ETAMAX);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", ETAMIN);
    
  if (etaMax_ > 44.5)
    {
      cout <<"<HcalBaseMonitor> WARNING:  etaMax_ value of "<<etaMax_<<" exceeds maximum allowed value of 44.5"<<endl;
      cout <<"                      Value being set back to 44.5."<<endl;
      cout <<"                      Additional code changes are necessary to allow value of "<<etaMax_<<endl;
      etaMax_ = 44.5;
    }

  if (etaMin_ < ETAMIN)
    {
      cout <<"<HcalBaseMonitor> WARNING:  etaMin_ value of "<<etaMin_<<" exceeds minimum allowed value of 44.5"<<endl;
      cout <<"                      Value being set back to -44.5."<<endl;
      cout <<"                      Additional code changes are necessary to allow value of "<<etaMin_<<endl;
      etaMin_ = -44.5;
    }

  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", PHIMAX);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", PHIMIN);
  phiBins_ = (int)(phiMax_ - phiMin_);

  return;
} //void HcalBaseMonitor::setup()

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



} // bool  HcalBaseMonitor::validDetId(HcalSubdetector sd, int ies, int ip, int dp)



// Create vectors of MonitorElements for individual depths


// *********************************************************** //

void HcalBaseMonitor::SetupEtaPhiHists(MonitorElement* &h, EtaPhiHists & hh, char* Name, char* Units)
{
  stringstream name;
  name<<Name;
  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  h=m_dbe->book2D(("All "+name.str()+unitname.str()).c_str(),
		  (name.str() + " for all HCAL ("+unittitle.str().c_str()+")"),
		  85,-42.5,42.5,
		  72,0.5,72.5);

  h->setAxisTitle("i#eta",1);
  h->setAxisTitle("i#phi",2);
  
  SetupEtaPhiHists(hh, Name, Units);
}

void HcalBaseMonitor::SetupEtaPhiHists(EtaPhiHists & hh, char* Name, char* Units)
{
  stringstream name;
  name<<Name;

  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  // Push back depth plots
  hh.depth.push_back(m_dbe->book2D(("HB HE HF Depth 1 "+name.str()+unitname.str()).c_str(),
		      (name.str()+" Depth 1 -- HB HE HF ("+unittitle.str().c_str()+")"),
		      85,-42.5,42.5,
		      72,0.5,72.5));
  hh.depth.push_back(m_dbe->book2D(("HB HE HF Depth 2 "+name.str()+unitname.str()).c_str(),
		      (name.str()+" Depth 2 -- HB HE HF ("+unittitle.str().c_str()+")"),
		      85,-42.5,42.5,
		      72,0.5,72.5));
  // Set up variable-sized bins for HE depth 3 (MonitorElement also requires phi bins to be entered in array format)
  float xbins[]={(float)-28.5,(float)-27.5,(float)-26.5,(float)-16.5,(float)-15.5,
		 (float)15.5,(float)16.5,(float)26.5,(float)27.5,(float)28.5};
  float ybins[73];
  for (int i=0;i<=72;i++) ybins[i]=(float)(i+0.5);
  hh.depth.push_back(m_dbe->book2D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
				   (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
				   // Use variable-sized eta bins 
				   9, xbins, 72, ybins));
  // HO bins are fixed width, but cover a smaller eta range (-15 -> 15)
  hh.depth.push_back(m_dbe->book2D(("HO Depth 4 "+name.str()+unitname.str()).c_str(),
				   (name.str()+" Depth 4 -- HO ("+unittitle.str().c_str()+")"),
				   31,-15.5,15.5,
				   72,0.5,72.5));
  hh.setBinLabels(); // set axis titles, special bins
  
}



void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units)
{
  /* Code makes overall 2D MonitorElement histogram,
     and the vector of 2D MonitorElements for each individual depth.
     Eta, Phi bins are set automatically from the etaMax_, etaMin_, etc.
     values in HcalBaseMonitor.h
  */

  /*
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  */
  stringstream name;
  name<<Name;
  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  h=m_dbe->book2D(("All "+name.str()+unitname.str()).c_str(),
		  (name.str() + " for all HCAL ("+unittitle.str().c_str()+")"),
		  etaBins_, etaMin_, etaMax_,
		  phiBins_, phiMin_, phiMax_);
  h->setAxisTitle("i#eta",1);
  h->setAxisTitle("i#phi",2);
  
  setupDepthHists2D(hh, Name, Units);
  /*
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }
  */
  return;
} // void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units)


// *************************************** //

void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, char* Name, char* Units)
{
  /* Code makes vector of 2D MonitorElements for all 6 depths
     (4 depths, + 2 for separate HE histograms).
     Bins are automatically set for eta/phi indices
  */
  
  /*
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  */
  stringstream name;
  name<<Name;

  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  // Push back depth plots
  hh.push_back(m_dbe->book2D(("HB HF Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HB & HF only ("+unittitle.str().c_str()+")"),
			     etaBins_,etaMin_,etaMax_,
			     phiBins_,phiMin_,phiMax_));
  hh.push_back( m_dbe->book2D(("HB HF Depth 2 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 2 -- HB & HF only ("+unittitle.str().c_str()+")"),
			      etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_));
  hh.push_back( m_dbe->book2D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
			      etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_));
  hh.push_back( m_dbe->book2D(("HO ZDC "+name.str()+unitname.str()).c_str(),
			      (name.str()+" -- HO & ZDC ("+unittitle.str().c_str()+")"),
			      etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_));
  hh.push_back(m_dbe->book2D(("HE Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HE only ("+unittitle.str().c_str()+")"),
			     etaBins_,etaMin_,etaMax_,
			     phiBins_,phiMin_,phiMax_));
  hh.push_back(m_dbe->book2D(("HE Depth 2 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 2 -- HE only ("+unittitle.str().c_str()+")"),
			     etaBins_,etaMin_,etaMax_,
			     phiBins_,phiMin_,phiMax_));
  for (unsigned int i=0;i<hh.size();++i)
    {
      hh[i]->setAxisTitle("i#eta",1);
      hh[i]->setAxisTitle("i#phi",2);
    }
  /* 
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }
  */
  return;
} // void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, char* Name, char* Units)


// *************************************************************** //

void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units, 
					int nbinsx, int lowboundx, int highboundx, 
					int nbinsy, int lowboundy, int highboundy)
{
  /* Code makes overall 2D MonitorElement histogram,
     and the vector of 2D MonitorElements for each individual depth.
     Bin ranges, sizes are specified by user
  */

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  stringstream name;
  name<<Name;
  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  h=m_dbe->book2D(("All "+name.str()+unitname.str()).c_str(),
                  (name.str() + " for all HCAL ("+unittitle.str().c_str()+")"),
		  nbinsx, lowboundx, highboundx,
		  nbinsy, lowboundy, highboundy);

  setupDepthHists2D(hh, Name, Units, 
		    nbinsx, lowboundx, highboundx,
		    nbinsy, lowboundy, highboundy);


  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units, int nbinsx...)

// *************************************************************** //


void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, char* Name, char* Units,
					int nbinsx, int lowboundx, int highboundx,
					int nbinsy, int lowboundy, int highboundy)
{
  /* Code makes vector of 2D MonitorElements for all 6 depths
     (4 depths, + 2 for separate HE histograms).
     Bins are automatically set for eta/phi indices
  */

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  stringstream name;
  name<<Name;

  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  // Push back depth plots
  hh.push_back(m_dbe->book2D(("HB HF Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HB & HF only ("+unittitle.str().c_str()+")"),
			     nbinsx, lowboundx, highboundx,
			     nbinsy, lowboundy, highboundy));
  hh.push_back( m_dbe->book2D(("HB HF Depth 2 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 2 -- HB & HF only ("+unittitle.str().c_str()+")"),
			      nbinsx, lowboundx, highboundx,
			      nbinsy, lowboundy, highboundy));
  hh.push_back( m_dbe->book2D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
			      nbinsx, lowboundx, highboundx,
			      nbinsy, lowboundy, highboundy));
  hh.push_back( m_dbe->book2D(("HO ZDC "+name.str()+unitname.str()).c_str(),
			      (name.str()+" -- HO & ZDC ("+unittitle.str().c_str()+")"),
			      nbinsx, lowboundx, highboundx,
			      nbinsy, lowboundy, highboundy));
  hh.push_back(m_dbe->book2D(("HE Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HE only ("+unittitle.str().c_str()+")"),
			     nbinsx, lowboundx, highboundx,
			     nbinsy, lowboundy, highboundy));
  hh.push_back(m_dbe->book2D(("HE Depth 2 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 2 -- HE only ("+unittitle.str().c_str()+")"),
			     nbinsx, lowboundx, highboundx,
			     nbinsy, lowboundy, highboundy));
 
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, char* Name, char* Units)

// ****************************************** //

void HcalBaseMonitor::setupDepthHists1D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units, int lowbound, int highbound, int Nbins)
{
  // Makes an overall 1D Monitor Element (for summing over depths) for h, and creates individual depth Monitor Elements for hh
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  stringstream name;
  name<<Name;

  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }
  
  // Create overall 1D Monitor Element
  h=m_dbe->book1D(("All "+name.str()+unitname.str()).c_str(),
		  (name.str() + " for all HCAL ("+unittitle.str().c_str()+")"),
		  Nbins,lowbound,highbound);
  h->setAxisTitle(unitname.str().c_str(),1);
  
  // Create vector of Monitor Elements for individual depths
  setupDepthHists1D(hh, Name, Units, lowbound, highbound, Nbins);

   if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS1D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }
   return;

} //void HcalBaseMonitor::setupDepthHists1D(MonitorElement* &h, std::vector<MonitorElement*> &hh, char* Name, char* Units)



void HcalBaseMonitor::setupDepthHists1D(std::vector<MonitorElement*> &hh, char* Name, char* Units, int lowbound, int highbound, int Nbins)
{
  // Makes histograms just for the vector of Monitor Elements
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  stringstream name;
  name<<Name;
  stringstream unitname;
  stringstream unittitle;
  if (Units=="")
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  // Push back depth plots
  hh.push_back(m_dbe->book1D(("HB HF Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HB & HF only ("+unittitle.str().c_str()+")"),
			     Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HB HF Depth 2 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 2 -- HB & HF only ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
			      (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HO ZDC "+name.str()+unitname.str()).c_str(),
			      (name.str()+" -- HO & ZDC ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));
  hh.push_back(m_dbe->book1D(("HE Depth 1 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 1 -- HE only ("+unittitle.str().c_str()+")"),
			     Nbins,lowbound,highbound));
  hh.push_back(m_dbe->book1D(("HE Depth 2 "+name.str()+unitname.str()).c_str(),
			     (name.str()+" Depth 2 -- HE only ("+unittitle.str().c_str()+")"),
			     Nbins,lowbound,highbound));

  for (unsigned int i=0;i<hh.size();++i)
    {
      hh[i]->setAxisTitle(unitname.str().c_str(),1);
    }
 
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS1D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalBaseMonitor::setupDepthHists1D(std::vector<MonitorElement*> &hh, char* Name, char* Units, int lowbound, int highbound, int Nbins)


void HcalBaseMonitor::setMinMaxHists2D(std::vector<MonitorElement*> &hh, double min, double max)
{
  for (unsigned int i=0; i<hh.size();++i)
    {
      TH2F* histo=hh[i]->getTH2F();
      histo->SetMinimum(min);
      histo->SetMaximum(max);
    }
  return;
}

void HcalBaseMonitor::setMinMaxHists1D(std::vector<MonitorElement*> &hh, double min, double max)
{
  for (unsigned int i=0; i<hh.size();++i)
    {
      TH1F* histo=hh[i]->getTH1F();
      histo->SetMinimum(min);
      histo->SetMaximum(max);
    }
  return;
}


void HcalBaseMonitor::FillUnphysicalHEHFBins(std::vector<MonitorElement*> &hh)
{
  // This fills in the regions of the eta-phi map where the HCAL phi segmentation is greater than 5 degrees.
  // This version does the fill for the "St. John 6" set of histograms

  if (!fillUnphysical_) return; 

  int ieta=0;
  int iphi=0;

  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      if (abs(ieta)<21)
	continue;
      for (int phi=0;phi<72;++phi)
	{
	  iphi=phi+1;
	  if (iphi%2==1 && abs(ieta)<40 && iphi<75 && iphi>0)
	    {
	      hh[0]->setBinContent(eta+2,phi+3,hh[0]->getBinContent(eta+2,phi+2));
	      hh[1]->setBinContent(eta+2,phi+3,hh[1]->getBinContent(eta+2,phi+2));
	      hh[2]->setBinContent(eta+2,phi+3,hh[2]->getBinContent(eta+2,phi+2));
	      hh[4]->setBinContent(eta+2,phi+3,hh[4]->getBinContent(eta+2,phi+2));
	      hh[5]->setBinContent(eta+2,phi+3,hh[5]->getBinContent(eta+2,phi+2));
	    } // if (iphi%2==1...)
	  else if (abs(ieta)>39 && iphi%4==3 && iphi<75)
	    {
	      // Set last two eta strips where each cell spans 20 degrees in phi
	      // Set next phi cell above iphi, and 2 cells below the actual cell 
	      hh[0]->setBinContent(eta+2,phi+3,hh[0]->getBinContent(eta+2,phi+2));
	      hh[0]->setBinContent(eta+2,phi+1,hh[0]->getBinContent(eta+2,phi+2));
	      hh[0]->setBinContent(eta+2,phi,hh[0]->getBinContent(eta+2,phi+2));
	      hh[1]->setBinContent(eta+2,phi+3,hh[1]->getBinContent(eta+2,phi+2));
	      hh[1]->setBinContent(eta+2,phi+1,hh[1]->getBinContent(eta+2,phi+2));
	      hh[1]->setBinContent(eta+2,phi,hh[1]->getBinContent(eta+2,phi+2));
	      
	    } // else if (abs(ieta)>39 ...)
	} // for (int phi=0;phi<72;++phi)
      
    } // for (int eta=0; eta< (etaBins_-2);++eta)
  return;
} // void HcalBaseMonitor::FillUnphysicalHEHFBins(std::vector<MonitorElement*> &hh)

void HcalBaseMonitor::FillUnphysicalHEHFBins(EtaPhiHists &hh)
{
  int ieta=0;
  int iphi=0;
  // First 2 depths have 5-10-20 degree corrections
  for (unsigned int d=0;d<3;++d)
    {
      for (int eta=0;eta<hh.depth[d]->getNbinsX();++eta)
	{
	  for (int phi=0;phi<hh.depth[d]->getNbinsY();++phi)
	    {
	      ieta=CalcIeta(eta,d);
	      if (ieta==-9999 || abs(ieta)<21) continue;
	      iphi=phi+1;
	      if (iphi%2==1 && abs(ieta)<40 && iphi<73)
		{
		  hh.depth[d]->setBinContent(eta+1,iphi+1,hh.depth[d]->getBinContent(eta+1,iphi));
		}
	      // last two eta strips span 20 degrees in phi
	      // Fill the phi cell above iphi, and the 2 below it
	      else  if (abs(ieta)>39 && iphi%4==3 && iphi<73)
		{
		  hh.depth[d]->setBinContent(eta+1,iphi+1, hh.depth[d]->getBinContent(eta+1,iphi));
		  hh.depth[d]->setBinContent(eta+1,iphi-1, hh.depth[d]->getBinContent(eta+1,iphi));
		  hh.depth[d]->setBinContent(eta+1,iphi-2, hh.depth[d]->getBinContent(eta+1,iphi));
		}
	    } // for (int phi...)
	} // for (int eta...)
    } // for (int d=0;...)
  // no corrections needed for HO (depth 4)
  return;
} // HcalBaseMonitor::HcalBaseMonitor::FillUnphysicalHEHFBins(MonitorElement* hh)



void HcalBaseMonitor::FillUnphysicalHEHFBins(MonitorElement* hh)
{
  // This fills in the regions of the eta-phi map where the HCAL phi segmentation is greater than 5 degrees.
  // This version does the fill for only a single MonitorElement
  // Do we want to be more careful here in the future, summing over the individual problem cells?

  int ieta=0;
  int iphi=0;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      if (abs(ieta)<21)
	continue;
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  if (iphi%2==1 && abs(ieta)<40 && iphi<75 && iphi>0)
	    {
	      hh->setBinContent(eta+2,phi+3,hh->getBinContent(eta+2,phi+2));
	    } // if (iphi%2==1...)
	  else if (abs(ieta)>39 && iphi%4==3 && iphi<75)
	    {
	      // Set last two eta strips where each cell spans 20 degrees in phi
	      // Set next phi cell above iphi, and 2 cells below the actual cell 
	      hh->setBinContent(eta+2,phi+3,hh->getBinContent(eta+2,phi+2));
	    } // else if (abs(ieta)>39 ...)
	} // for (int phi=0;phi<72;++phi)

    } // for (int eta=0; eta< (etaBins_-2);++eta)

  return;
} // HcalBaseMonitor::FillUnphysicalHEHFBins(std::vector<MonitorElement*> &hh)


int HcalBaseMonitor::CalcEtaBin(int subdet, int ieta, int depth)
{
  // This takes the eta value from a subdetector and return an eta counter value as used by eta-phi histograms
  // (ieta=-41 corresponds to bin 0, +41 to bin 85 -- there are two offsets to deal with the overlap at |ieta|=29).
  // For HO, ieta = -15 corresponds to bin 0, and ieta=15 is bin 30
  // For HE depth 3, things are more complicated, but feeding the ieta value will give back the corresponding counter eta value
  int etabin=-9999; 
  if (depth<=2)
    {
      etabin=ieta+42;
      if (subdet==HcalForward)
	{
	  ieta < 0 ? etabin-- : etabin++;
	}
    }
  else if (subdet==HcalOuter && abs(ieta)<16)
    etabin=ieta+15;
  else if (subdet==HcalEndcap)
    {
      if (depth==3)
	{
	  if (ieta==-28) etabin=0;
	  else if (ieta==-27) etabin=1;
	  else if (ieta==-16) etabin=3;
	  else if (ieta==16)  etabin=5;
	  else if (ieta==27)  etabin=7;
	  else if (ieta==28)  etabin=8;
	}
    }
  return etabin;
}

int HcalBaseMonitor::CalcIeta(int subdet, int eta, int depth)
{
  int ieta;
  ieta=eta-42; // default shift: bin 0 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
  if (subdet==HcalBarrel)
    {
      if (depth>2) 
	ieta=-9999; // non-physical value
    }
  else if (subdet==HcalForward)
    {
      if (depth>2)
	ieta=-9999;
      else
	{
	  if (eta<14) ieta++;
	  else if (eta>72) ieta--;
	  else ieta=-9999; // if outside forward range, return dummy
	}
    }
  // add in HE depth 3, HO later
  else if (subdet==HcalEndcap)
    {
      if (depth==3)
	{
	  if (eta<0 || eta>8) ieta=-9999;
	  else
	    {
	      int binmap[]={-28,-27,-9999,-16,-9999,16,9999,27,28};
	      ieta=binmap[eta];
	    }
	}
      else if (depth==4)
	ieta=-9999;
    } // HcalEndcap
  else if ( subdet==HcalOuter)
    {
      if (depth<4)
	ieta=-9999;
      else
	{
	  ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
	  if (abs(ieta)>15) ieta=-9999;
	}
    } // HcalOuter
  return ieta;
}
  
int HcalBaseMonitor::CalcIeta(int eta, int depth)
{
  int ieta;
  ieta=eta-42; // default shift: bin 0 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
  if (depth<=2)
    {
      if (eta<14) ieta++;
      else if (eta>72) ieta--;
    }
  else if (depth==3)
    {
      if (eta<0 || eta>8) ieta=-9999;
      else
	{
	  int binmap[]={-28,-27,-9999,-16,-9999,16,-9999,27,28};
	  ieta=binmap[eta];
	}
    }
  else if (depth==4)
    {
      ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
      if (abs(ieta)>15) ieta=-9999;
    }
  return ieta;
}

