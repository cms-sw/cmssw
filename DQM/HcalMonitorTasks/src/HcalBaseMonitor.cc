#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#define ETAMAX 44.5
#define ETAMIN -44.5
#define PHIMAX 73.5
#define PHIMIN -0.5


HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
  badCells_.clear();
  rootFolder_ = "Hcal";
  baseFolder_ = "BaseMonitor";
}

HcalBaseMonitor::~HcalBaseMonitor() {}

void HcalBaseMonitor::beginRun(){ievt_=0;levt_=0;LBprocessed_=false;}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  std::vector<std::string> dummy;
  dummy.clear();
  Online_   =  ps.getUntrackedParameter<bool>("Online",false);
  badCells_ =  ps.getUntrackedParameter<std::vector<std::string> >( "BadCells" , dummy);
  AllowedCalibTypes_ = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  

  // Base folder for the contents of this job
  std::string subsystemname = ps.getUntrackedParameter<std::string>("subSystemFolder", "Hcal") ;
  rootFolder_ = subsystemname + "/";

  // Global cfgs
  
  fVerbosity      = ps.getUntrackedParameter<int>("debug",0); 
  makeDiagnostics = ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);
  showTiming      = ps.getUntrackedParameter<bool>("showTiming",false);
  dump2database   = ps.getUntrackedParameter<bool>("dump2database",false); // dumps output to database file 

  checkHB_ = ps.getUntrackedParameter<bool>("checkHB",true);
  checkHE_ = ps.getUntrackedParameter<bool>("checkHE",true);
  checkHO_ = ps.getUntrackedParameter<bool>("checkHO",true);
  checkHF_ = ps.getUntrackedParameter<bool>("checkHF",true);
 
  checkNevents_ = ps.getUntrackedParameter<int>("checkNevents",1000); // specify how often to run checks
  Nlumiblocks_ = ps.getUntrackedParameter<int>("Nlumiblocks",1000); //  number of luminosity blocks to include in time plots 
  if (Nlumiblocks_<=0) Nlumiblocks_=1000;
  resetNevents_ = ps.getUntrackedParameter<int>("resetNevents",-1); // how often to reset histograms

  // Minimum error rate that will caused the problem histogram to be filled
  minErrorFlag_ = ps.getUntrackedParameter<double>("minErrorFlag",0.05);
  
  // Set eta, phi boundaries
  // We can remove these variables once we move completely to EtaPhiHists from the old 2D SJ6hists
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", ETAMAX);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", ETAMIN);
    
  if (etaMax_ > 44.5)
    {
      std::cout <<"<HcalBaseMonitor> WARNING:  etaMax_ value of "<<etaMax_<<" exceeds maximum allowed value of 44.5"<<std::endl;
      std::cout <<"                      Value being set back to 44.5."<<std::endl;
      std::cout <<"                      Additional code changes are necessary to allow value of "<<etaMax_<<std::endl;
      etaMax_ = 44.5;
    }

  if (etaMin_ < ETAMIN)
    {
      std::cout <<"<HcalBaseMonitor> WARNING:  etaMin_ value of "<<etaMin_<<" exceeds minimum allowed value of 44.5"<<std::endl;
      std::cout <<"                      Value being set back to -44.5."<<std::endl;
      std::cout <<"                      Additional code changes are necessary to allow value of "<<etaMin_<<std::endl;
      etaMin_ = -44.5;
    }

  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", PHIMAX);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", PHIMIN);
  phiBins_ = (int)(phiMax_ - phiMin_);

  ProblemsVsLB=0;
  ProblemsVsLB_HB=0;
  ProblemsVsLB_HE=0;
  ProblemsVsLB_HO=0;
  ProblemsVsLB_HF=0;
  
  meEVT_=0;
  meTOTALEVT_=0;
  ievt_=0;
  levt_=0;
  tevt_=0;
  lumiblock=0;
  oldlumiblock=0;
  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;

  return;
} //void HcalBaseMonitor::setup()

void HcalBaseMonitor::processEvent()
{
  // increment event counters
  ++ievt_;
  ++levt_;
  ++tevt_;
  // Fill MonitorElements
  if (m_dbe)
    {
      if (meEVT_) meEVT_->Fill(ievt_);
      if (meTOTALEVT_) meTOTALEVT_->Fill(tevt_);
    }
  return;
}

void HcalBaseMonitor::LumiBlockUpdate(int lb)
{

  //if (Online_ && lb<=lumiblock) // protect against mis-ordered LBs.  Handle differently in the future?
  //  return;

  lumiblock=lb;
  if (lumiblock==0) // initial lumiblock call; don't fill histograms, because nothing has been determined yet
      return;

  if (lb>Nlumiblocks_) // don't fill plots if lumiblock is beyond range
      return;

  // The following function would let us 'fill in' missing lumiblock sections.  
  // I think we only want this for Online running, since offline should fill each lumi block
  // independently.  
  // Should probably just do this in the individual tasks?
  /*
  if (Online_ && lumiblock<lb)
    {
      for (int i=lumiblock+1;i<lb;++i)
	{
	  if (ProblemsVsLB)
	    ProblemsVsLB->Fill(i,NumBadHB+NumBadHE+NumBadHO+NumBadHF);
	  if (ProblemsVsLB_HB)
	    ProblemsVsLB_HB->Fill(i,NumBadHB);
	  if (ProblemsVsLB_HE)
	    ProblemsVsLB_HE->Fill(i,NumBadHE);
	  if (ProblemsVsLB_HO)
	    ProblemsVsLB_HO->Fill(i,NumBadHO);
	  if (ProblemsVsLB_HF)
	    ProblemsVsLB_HF->Fill(i,NumBadHF);
	}
    }
  */
  return;
}

void HcalBaseMonitor::beginLuminosityBlock(int lumisec)
{
  // Protect against online mis-ordering of events.  
  // Do we want this enabled here? 
  //if (Online_ && lumisec<lumiblock)
  //  return;
  LumiBlockUpdate(lumisec);
  levt_=0;
  LBprocessed_=false;
} // beginLuminosityBlock

void HcalBaseMonitor::endLuminosityBlock()
{
  LBprocessed_=true;
  return;
} // endLuminosityBlock;

void HcalBaseMonitor::done(){}

void HcalBaseMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();    
    /*
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    m_dbe->removeContents();
    
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    m_dbe->removeContents();
    */
  }
  return;
} // void HcalBaseMonitor::clearME();

// ***************************************************************** //


bool HcalBaseMonitor::vetoCell(HcalDetId& id)
{
  /*
    Function identifies whether cell with HcalDetId 'id' should be vetoed, 
    based on elements stored in  badCells_ array.
  */

  if(badCells_.size()==0) return false;
  for(unsigned int i = 0; i< badCells_.size(); ++i)
    {
      
      unsigned int badc = atoi(badCells_[i].c_str());
      if(id.rawId() == badc) return true;
    }
  return false;
} // bool HcalBaseMonitor::vetoCell(HcalDetId id)

void HcalBaseMonitor::hideKnownBadCells()
{
  /* This prevents known bad cells from being displayed in overall problem maps and 
     depth histograms.  Is this what we want?  Or do we want some problems to be
     displayed in the depth plots but not the overall map?  (Or vice versa?)
  */
  
  for (unsigned int i=0;i<badCells_.size();++i)
    {
      unsigned int badc = atoi(badCells_[i].c_str());
      HcalDetId id(badc);
      int etabin=CalcEtaBin(id.subdet(),id.ieta(),id.depth());
      if (ProblemCells!=0) ProblemCells->setBinContent(etabin+1,id.iphi(),0);
      if (ProblemCellsByDepth.depth[id.depth()-1]!=0)
	ProblemCellsByDepth.depth[id.depth()-1]->setBinContent(etabin+1,id.iphi(),0);
    } // for (unsigned int i=0;...)
  return;
} // void HcalBaseMonitor::hideKnownBadCells()


// ************************************************************************************************************ //


// Create vectors of MonitorElements for individual depths

// *********************************************************** //

void HcalBaseMonitor::SetupEtaPhiHists(MonitorElement* &h, EtaPhiHists & hh, std::string Name, std::string Units)
{
  std::stringstream name;
  name<<Name;
  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
  return;
}

void HcalBaseMonitor::SetupEtaPhiHists(EtaPhiHists & hh, std::string Name, std::string Units)
{
  hh.setup(m_dbe, Name, Units);
  return;
}

void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units)
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
  std::stringstream name;
  name<<Name;
  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
      cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  */
  return;
} // void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units)


// *************************************** //

void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units)
{
  /* Code makes vector of 2D MonitorElements for all 6 depths
     (4 depths, + 2 for separate HE histograms).
     Bins are automatically set for eta/phi indices
     DEPRECATE THIS ONCE ALL OLD-STYLE 2D HISTOGRAMS HAVE BEEN REMOVED!
  */
  
  /*
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  */
  std::stringstream name;
  name<<Name;

  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
    {
      unitname<<Units;
      unittitle<<"No Units";
    }
  else
    {
      unitname<<" "<<Units;
      unittitle<<Units;
    }

  // Push back depth plots -- remove ZDC names at some point?
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
     cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  */
  return;
} // void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units)


// *************************************************************** //

void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units, 
					int nbinsx, int lowboundx, int highboundx, 
					int nbinsy, int lowboundy, int highboundy)
{
  /* Code makes overall 2D MonitorElement histogram,
     and the vector of 2D MonitorElements for each individual depth.
     Bin ranges, sizes are specified by user
     DEPRECATE THIS ONCE ALL OLD-STYLE 2D HISTOGRAMS HAVE BEEN REMOVED!
  */

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  std::stringstream name;
  name<<Name;
  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
      cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalBaseMonitor::setupDepthHists2D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units, int nbinsx...)

// *************************************************************** //


void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units,
					int nbinsx, int lowboundx, int highboundx,
					int nbinsy, int lowboundy, int highboundy)
{
  /* Code makes vector of 2D MonitorElements for all 6 depths
     (4 depths, + 2 for separate HE histograms).
     Bins are automatically set for eta/phi indices
     DEPRECATE THIS ONCE ALL OLD-STYLE 2D HISTOGRAMS HAVE BEEN REMOVED!
     
     
  */

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  std::stringstream name;
  name<<Name;

  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
      cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS2D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalBaseMonitor::setupDepthHists2D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units)

// ****************************************** //

void HcalBaseMonitor::setupDepthHists1D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units, int lowbound, int highbound, int Nbins)
{
  // Makes an overall 1D Monitor Element (for summing over depths) for h, and creates individual depth Monitor Elements for hh
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  std::stringstream name;
  name<<Name;

  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
      cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS1D_OVERALL "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }
   return;

} //void HcalBaseMonitor::setupDepthHists1D(MonitorElement* &h, std::vector<MonitorElement*> &hh, std::string Name, std::string Units)



void HcalBaseMonitor::setupDepthHists1D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units, int lowbound, int highbound, int Nbins)
{
  // Makes histograms just for the vector of Monitor Elements
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  std::stringstream name;
  name<<Name;
  std::stringstream unitname;
  std::stringstream unittitle;
  if (Units.empty())
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
  hh.push_back(m_dbe->book1D(("HB "+name.str()+unitname.str()).c_str(),
			     (name.str()+" HB ("+unittitle.str().c_str()+")"),
			     Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HE "+name.str()+unitname.str()).c_str(),
			      (name.str()+" HE ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HO "+name.str()+unitname.str()).c_str(),
			      (name.str()+" HO ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));
  hh.push_back( m_dbe->book1D(("HF "+name.str()+unitname.str()).c_str(),
			      (name.str()+" HF ("+unittitle.str().c_str()+")"),
			      Nbins,lowbound,highbound));

  for (unsigned int i=0;i<hh.size();++i)
    {
      hh[i]->setAxisTitle(unitname.str().c_str(),1);
    }
 
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalBaseMonitor SETUPDEPTHHISTS1D "<<name.str().c_str()<<" -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalBaseMonitor::setupDepthHists1D(std::vector<MonitorElement*> &hh, std::string Name, std::string Units, int lowbound, int highbound, int Nbins)


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

void HcalBaseMonitor::periodicReset()
{
  // when called, reset all counters, and all histograms
  if (ProblemCells!=0) ProblemCells->Reset();
  for (unsigned int i=0;i<ProblemCellsByDepth.depth.size();++i)
    {
      if (ProblemCellsByDepth.depth[i]!=0) 
	ProblemCellsByDepth.depth[i]->Reset();
    }
  return;
}
