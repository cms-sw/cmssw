#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"

#define ETAMAX 44.5
#define ETAMIN -44.5
#define PHIMAX 73.5
#define PHIMIN -0.5

using namespace cms;
using namespace edm;
using namespace std;



HcalSummaryClient::HcalSummaryClient() {} //constructor

void HcalSummaryClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<int>("debug", 0);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "Hcal");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // Find out which subtasks are being run
  dataFormatMon_.onoff=(ps.getUntrackedParameter<bool>("DataFormatClient",false));
  digiMon_.onoff=(ps.getUntrackedParameter<bool>("DigiClient",false));
  recHitMon_.onoff=(ps.getUntrackedParameter<bool>("RecHitClient",false));
  pedestalMon_.onoff=(ps.getUntrackedParameter<bool>("PedestalClient",false));
  pedestalMon_.onoff=false; // don't include pedestal monitoring in overall data quality?
  //ledMon_.onoff=(ps.getUntrackedParameter<bool>("LEDClient",false));
  hotCellMon_.onoff=(ps.getUntrackedParameter<bool>("HotCellClient",false));
  deadCellMon_.onoff=(ps.getUntrackedParameter<bool>("DeadCellClient",false));
  trigPrimMon_.onoff=(ps.getUntrackedParameter<bool>("TrigPrimClient",false));
  caloTowerMon_.onoff=(ps.getUntrackedParameter<bool>("CaloTowerClient",false));

  if (dataFormatMon_.onoff)
    dataFormatMon_.Setup("DataFormatMonitor",  // directory where problem depth histograms stored
			 " Hardware Watch Cells", // base name of depth histograms
			 "DataFormatMonitor/ HardwareWatchCells", // 
			 "DataFormat",
			 0.0);
  
  if (digiMon_.onoff)
    digiMon_.Setup("DigiMonitor_Hcal/problem_digis",
		   " Problem Digi Rate",
		   "DigiMonitor_Hcal/ ProblemDigis",
		   "Digi",
		   0.0);
  if (recHitMon_.onoff)
    recHitMon_.Setup("RecHitMonitor_Hcal/problem_rechits",
		     " Problem RecHit Rate",
		     "RecHitMonitor_Hcal/ ProblemRecHits",
		     "RecHit",
		     ps.getUntrackedParameter<double>("RecHitClient_minErrorFlag",0.0));
  if (pedestalMon_.onoff)
    pedestalMon_.Setup("BaselineMonitor_Hcal/problem_pedestals",
		       " Problem Pedestal Rate",
		       "BaselineMonitor_Hcal/ ProblemPedestals",
		       "PedEstimate",
		       ps.getUntrackedParameter<double>("PedestalClient_minErrorFlag",0.05));
		       
  if (hotCellMon_.onoff)
    hotCellMon_.Setup("HotCellMonitor_Hcal/problem_hotcells",
		      " Problem Hot Cell Rate",
		      "HotCellMonitor_Hcal/ ProblemHotCells",
		      "HotCell",
		      ps.getUntrackedParameter<double>("HotCellClient_minErrorFlag",0.05));
  if (deadCellMon_.onoff)
    deadCellMon_.Setup("DeadCellMonitor_Hcal/problem_deadcells",
		       " Problem Dead Cell Rate",
		       "DeadCellMonitor_Hcal/ ProblemDeadCells",
		       "DeadCell",
		       ps.getUntrackedParameter<double>("DeadCellClient_minErrorFlag",0.05));

  // All initial status floats set to -1 (unknown)
  // status floats give fraction of good cells in detector
  status_HB_=-1;
  status_HE_=-1;
  status_HO_=-1;
  status_HF_=-1;
  status_ZDC_=-1;
  status_global_=-1;
  
  // set total number of cells in each subdetector
  subdetCells_.insert(make_pair("HB",2592));
  subdetCells_.insert(make_pair("HE",2592));
  subdetCells_.insert(make_pair("HO",2160));
  subdetCells_.insert(make_pair("HF",1728));
  subdetCells_.insert(make_pair("ZDC",18));

  // Assume subdetectors absent at start
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;
  ZDCpresent_=0;

} // HcalSummaryClient::HcalSummaryClient(const ParameterSet& ps)

HcalSummaryClient::~HcalSummaryClient()
{
} //destructor

void HcalSummaryClient::beginJob(DQMStore* dqmStore)
{
  dqmStore_=dqmStore;
  if (debug_>0) 
    std::cout <<"<HcalSummaryClient: beginJob>"<<std::endl;
  ievt_ = 0; // keepts track of all events in job
  jevt_ = 0; // keeps track of all events in run
  lastupdate_=0; // keeps analyze from being called by both endRun and endJob
} // void HcalSummaryClient::beginJob(DQMStore* dqmStore)

void HcalSummaryClient::beginRun(void)
{
  if ( debug_>0 ) std::cout << "<HcalSummaryClient: beginRun>" << std::endl;

  jevt_ = 0;
  this->setup();
} //void HcalSummaryClient::beginRun(void)

void HcalSummaryClient::endJob(void)
{
  if ( debug_>0 ) std::cout << "<HcalSummaryClient: endJob> ievt = " << ievt_ << std::endl;
  // When the job ends, do we want to make a summary before exiting?
  // Or does this interfere with normalization of histograms?
  //if (ievt_>lastupdate_)
  //  analyze();
  this->cleanup();
} // void HcalSummaryClient::endJob(void)

void HcalSummaryClient::endRun(void) 
{
  if ( debug_ ) std::cout << "<HcalSummaryClient: endRun> jevt = " << jevt_ << std::endl;
  // When the run ends, do we want to make a summary before exiting?
  // Or does this interfere with normalization of histograms?
  //analyze();
  lastupdate_=ievt_;
  this->cleanup();
} // void HcalSummaryClient::endRun(void) 

void HcalSummaryClient::setup(void)
{
  MonitorElement* me;
  ostringstream histo;
  // set overall status
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo");
  histo<<"reportSummary";
  me=dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  me = dqmStore_->bookFloat(histo.str().c_str());
  me->Fill(-1); // set status to unknown at startup
  histo.str("");

  std::string subdets[5] = {"HB","HE","HO","HF","ZDC"};
  for (unsigned int i=0;i<5;++i)
    {
      // Create floats showing subtasks status
      dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );  
      histo<<"Hcal_"<<subdets[i].c_str();
      me=dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
      if (me)
	dqmStore_->removeElement(me->getName());
      me = dqmStore_->bookFloat(histo.str().c_str());
      me->Fill(-1); // set status to unknown at startup
      histo.str("");
    }

  // Make depth 2D histograms
  dqmStore_->setCurrentFolder(prefixME_+"/EventInfo/");

  histo<<"HB HE HF Depth 1 Summary Map";
  me = dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  histo<<"HB HE HF Depth 2 Summary Map";
  me = dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  histo<<"HE Depth 3 Summary Map";
  me = dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  histo<<"HO Depth 4 Summary Map";
  me = dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());

  EtaPhiHists SummaryMapByDepth;
  SummaryMapByDepth.setup(dqmStore_,"Summary Map");
  // Set histogram values to -1
  // Set all bins to "unknown" to start
  int etabins=0;
  for (unsigned int depth=0;depth<4;++depth)
    {
      etabins=SummaryMapByDepth.depth[depth]->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    SummaryMapByDepth.depth[depth]->setBinContent(ieta+1,iphi+1,-1);
	}
    }
  
  // Make new simplified status histogram
  histo.str("");
  histo<<"reportSummaryMap";
  me=dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  me = dqmStore_->book2D(histo.str().c_str(), histo.str().c_str(), 
			 5,0,5,1,0,1);
  TH2F* myhist=me->getTH2F();
  myhist->GetXaxis()->SetBinLabel(1,"HB");
  myhist->GetXaxis()->SetBinLabel(2,"HE");
  myhist->GetXaxis()->SetBinLabel(3,"HO");
  myhist->GetXaxis()->SetBinLabel(4,"HF");
  myhist->GetYaxis()->SetBinLabel(1,"Status");
  // Add ZDC at some point
  myhist->GetXaxis()->SetBinLabel(5,"ZDC");
  myhist->SetBinContent(5,1,-1); // no ZDC info known
  myhist->SetOption("textcolz");
  //myhist->SetOptStat(0);

  // Set initial counters to -1 (unknown)
  status_global_=-1; 
  status_HB_=-1; 
  status_HE_=-1; 
  status_HO_=-1; 
  status_HF_=-1; 
  status_ZDC_=-1;

  // Set all bins to "unknown" to start
  if(dqmStore_->get(prefixME_+"/EventInfo/HB HE HF Depth 1 Summary Map"))
    {
      depthME.push_back(dqmStore_->get(prefixME_+"/EventInfo/HB HE HF Depth 1 Summary Map"));
      (depthME[depthME.size()-1]->getTH2F())->SetMaximum(1);
      (depthME[depthME.size()-1]->getTH2F())->SetMinimum(-1);
      int etabins=depthME[depthME.size()-1]->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    depthME[depthME.size()-1]->setBinContent(ieta+1,iphi+1,-1);
	}
    }

  if(dqmStore_->get(prefixME_+"/EventInfo/HB HE HF Depth 2 Summary Map"))
    {
      depthME.push_back(dqmStore_->get(prefixME_+"/EventInfo/HB HE HF Depth 2 Summary Map"));
      (depthME[depthME.size()-1]->getTH2F())->SetMaximum(1);
      (depthME[depthME.size()-1]->getTH2F())->SetMinimum(-1);
      int etabins=depthME[depthME.size()-1]->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    depthME[depthME.size()-1]->setBinContent(ieta+1,iphi+1,-1);
	}
    }

  if(dqmStore_->get(prefixME_+"/EventInfo/HE Depth 3 Summary Map"))
    {
      depthME.push_back(dqmStore_->get(prefixME_+"/EventInfo/HE Depth 3 Summary Map"));
      (depthME[depthME.size()-1]->getTH2F())->SetMaximum(1);
      (depthME[depthME.size()-1]->getTH2F())->SetMinimum(-1);
      int etabins=depthME[depthME.size()-1]->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    depthME[depthME.size()-1]->setBinContent(ieta+1,iphi+1,-1);
	}
    }
  if(dqmStore_->get(prefixME_+"/EventInfo/HO Depth 4 Summary Map"))
    {
      depthME.push_back(dqmStore_->get(prefixME_+"/EventInfo/HO Depth 4 Summary Map"));
      (depthME[depthME.size()-1]->getTH2F())->SetMaximum(1);
      (depthME[depthME.size()-1]->getTH2F())->SetMinimum(-1);
      int etabins=depthME[depthME.size()-1]->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    depthME[depthME.size()-1]->setBinContent(ieta+1,iphi+1,-1);
	}
    }

  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (!reportMap)
    {
      std::cout <<"<HcalSummaryClient::setup> Could not get reportSummaryMap!"<<std::endl;
      return;
    }
  for (int i=1;i<=5;++i)
    reportMap->setBinContent(i,1,-1);

  return;
      
} // void HcalSummaryClient::setup(void)


void HcalSummaryClient::cleanup(void) 
{
  
  if ( ! enableCleanup_ ) return;

  MonitorElement* me;

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary"); 
  if (me)
    {
      dqmStore_->removeElement(me->getName());
    }
} // void HcalSummaryClient::cleanup(void)


void HcalSummaryClient::incrementCounters(void)
{
  ++ievt_;
  ++jevt_;
  return;
} // void HcalSummaryClient::incrementCounters()


void HcalSummaryClient::analyze(void)
{
  if (debug_>0)
    std::cout <<"<HcalSummaryClient::analyze>  Running analyze..."<<std::endl;
  if ( ievt_ % 10 == 0 ) 
    {
      if ( debug_>1 )
	std::cout << "<HcalSummaryClient::analyze> ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
    }

  // Reset summary map to 'unknown' status 

  MonitorElement* simpleMap = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (!simpleMap)
    {
      std::cout <<"<HcalSummaryClient::analyze> Could not get reportSummaryMap!"<<std::endl;
      return;
    }
  for (int ix=1;ix<=5;++ix)
    simpleMap->setBinContent(ix,1,-1);

  // reset all depth histograms

  // Start with counters in 'unknown' status; they'll be set by analyze_everything routines 
  status_global_=-1; 
  status_HB_=-1; 
  status_HE_=-1; 
  status_HO_=-1; 
  status_HF_=-1; 
  status_ZDC_=-1;

  // check to find which subdetectors are present
  MonitorElement* temp_present;
  if (HBpresent_==0)
    {
      temp_present = dqmStore_->get(prefixME_+"/DQM Job Status/HBpresent");
      if (temp_present)
	HBpresent_=temp_present->getIntValue();
    }
  if (HEpresent_==0)
    {
      temp_present = dqmStore_->get(prefixME_+"/DQM Job Status/HEpresent");
      if (temp_present)
	HEpresent_=temp_present->getIntValue();
    }
  if (HOpresent_==0)
    {
      temp_present = dqmStore_->get(prefixME_+"/DQM Job Status/HOpresent");
      if (temp_present)
	HOpresent_=temp_present->getIntValue();
    }
  if (HFpresent_==0)
    {
      temp_present = dqmStore_->get(prefixME_+"/DQM Job Status/HFpresent");
      if (temp_present)
	HFpresent_=temp_present->getIntValue();
    }
 /*
   // not yet ready for ZDC checking
 if (ZDCpresent_==0)
    {
      temp_present = dqmStore_->get(prefixME_+"/DQM Job Status/ZDCpresent");
      if (temp_present)
	ZDCpresent_=temp_present->getIntValue();
    }
 */


 if (debug_>1) 
   std::cout <<"<HcalSummaryClient::analyze>  HB present = "<<HBpresent_<<" "<<"HE present = "<<HEpresent_<<" "<<"HO present = "<<HOpresent_<<" "<<"HF present = "<<HFpresent_<<" ZDC present = "<<ZDCpresent_<<std::endl;

 // set status to 0 if subdetector is present
 if (HBpresent_) status_HB_=0;
 if (HEpresent_) status_HE_=0;
 if (HOpresent_) status_HO_=0;
 if (HFpresent_) status_HF_=0;
 if (ZDCpresent_) status_ZDC_=0;
 if (HBpresent_ || HEpresent_ || HOpresent_ || HFpresent_ ) // don't include ZDC yet
   status_global_=0;

 // Set starting histogram values to 0
 resetSummaryPlots();

 // Calculate status values for individual tasks
 if (dataFormatMon_.IsOn()) analyze_subtask(dataFormatMon_);
 if (digiMon_.IsOn()) analyze_subtask(digiMon_);
 if (recHitMon_.IsOn()) analyze_subtask(recHitMon_);
 if (pedestalMon_.IsOn()) analyze_subtask(pedestalMon_);
 if (ledMon_.IsOn()) analyze_subtask(ledMon_);
 if (hotCellMon_.IsOn()) analyze_subtask(hotCellMon_);
 if (deadCellMon_.IsOn()) analyze_subtask(deadCellMon_);
 if (trigPrimMon_.IsOn()) analyze_subtask(trigPrimMon_);
 if (caloTowerMon_.IsOn()) analyze_subtask(caloTowerMon_);

 // Okay, we've got the individual tasks; now form the combined value

 int ieta=-9999;
 for (unsigned int d=0;d<depthME.size();++d)
   {
     for (int eta=0;eta<depthME[d]->getNbinsX();++eta)
       {
	 ieta=CalcIeta(eta,d+1);
	 if (ieta==-9999) continue;
	 for (int phi=0;phi<72;++phi)
	   {
	     // skip unphysical values
	     if (abs(ieta)>20 && abs(ieta)<40 && (phi+1)%2!=1) continue;
	     if (abs(ieta)>39 && (phi+1)%4!=3) continue;
	     if (depthME[d]->getBinContent(eta+1,phi+1)>0)
	       {
		 if (isHB(eta,d+1)) ++status_HB_;
		 else if (isHE(eta,d+1)) ++status_HE_;
		 else if (isHO(eta,d+1)) ++status_HO_;
		 else if (isHF(eta,d+1)) ++status_HF_;
	       }
	   } // loop over phi
       } // loop over eta
   } //loop over depth

  int totalcells=0;
  std::map<std::string, int>::const_iterator it;

  // need to loop over report summary plots
  if (HBpresent_)
    {
      status_global_+=status_HB_; 
      it=subdetCells_.find("HB");
      totalcells+=it->second;
      status_HB_= 1-(status_HB_/it->second);
      status_HB_=max(0.,status_HB_); // converts fraction of bad channels to good fraction
    }
  if (HEpresent_)
    {
      status_global_+=status_HE_;
      it=subdetCells_.find("HE");
      totalcells+=it->second;
      status_HE_= 1-(status_HE_/it->second);
      status_HE_=max(0.,status_HE_); // converts fraction of bad channels to good fraction
    }

  if (HOpresent_)
    {
      status_global_+=status_HO_;
      it=subdetCells_.find("HO");
      totalcells+=it->second;
      status_HO_= 1-(status_HO_/it->second);
      status_HO_=max(0.,status_HO_); // converts fraction of bad channels to good fraction
    }
  if (HFpresent_)
    {
      status_global_+=status_HF_;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      status_HF_= 1-(status_HF_/it->second);
      status_HF_=max(0.,status_HF_); // converts fraction of bad channels to good fraction
    }
  /*
 if (ZDCpresent_)
    {
    status_global_+=status_ZDC_;
    it=subdetCells_.find("ZDC");
    totalcells+=it->second;
    status_ZDC_= 1-(status_ZDC_/it->second);
    status_ZDC_=max(0.,status_ZDC_); // converts fraction of bad channels to good fraction
    }
  */
  
  if (totalcells==0)
    status_global_=-1;
  else
    {
      status_global_=1-status_global_/totalcells;
      status_global_=max(0.,status_global_); // convert to good fraction
      // Now loop over cells in reportsummarymap, changing from bad fraction to good

      int ieta,iphi;

      for (unsigned int depth=0;
	   depth<depthME.size();
	   ++depth)
      {
	int etaBins=depthME[depth]->getNbinsX();
	double reportval=0;

	for (int eta=0;eta<etaBins;++eta)
	  {
	    ieta=CalcIeta(eta, 1);
	    if (ieta==-9999) continue;
	    for (int phi=0; phi<72;++phi)
	      {
		reportval=depthME[depth]->getBinContent(eta+1,phi+1);
		if (reportval>-1)
		  {
		    iphi=phi+1;
		    /*
		      if (abs(ieta)>20 && iphi%2!=1)
		      continue;
		    if (abs(ieta)>39 &&iphi%4!=3)
		      continue;
		    */
		    depthME[depth]->setBinContent(eta+1,phi+1,max(0.,1-reportval));
		    
		    /*
		    if (fillUnphysical_)
		      {
			if (depth==3) continue; // skip HO
			// fill even phi cells in region where cells span 10 degrees in phi
			// ("True" cell values are phi=1,3,5,7,...) 
			if (abs(ieta)>20 && abs(ieta)<40 && iphi%2==1)
			  {
			    depthME[depth]->setBinContent(eta+1,phi+2,reportval);
			  }
			
			// fill all cells in region where cells span 20 degrees in phi
			// (actual cell phi values are 3,7,11,...)
			else if (abs(ieta)>39 && phi%4==3)
			  {
			    depthME[depth]->setBinContent(eta,phi+2,reportval);
			    depthME[depth]->setBinContent(eta,phi,reportval);
			    depthME[depth]->setBinContent(eta,phi-1,reportval);
			  }
		      }
		    */
		  } //if (bincontent>-1)
	      } // for (int phi=0;...)
	  } // for (int eta=0;...)
      } // for (int d=0;...)
    } // else (totalcells>0)

  // Now set the status words
  MonitorElement* me;
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );
  

  if (debug_>1) std::cout <<"SUMMARY = "<<status_HB_<<"\t"<<status_HE_<<"\t"<<status_HO_<<"\t"<<status_HF_<<"\t"<<status_global_<<std::endl;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me) 
    {
      me->Fill(status_global_);
      //simpleMap->setBinContent(5,1,status_global_);
    }

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HB");
  if (me)
    {
      me->Fill(status_HB_);
      simpleMap->setBinContent(1,1,status_HB_);
    }
  
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HE");
  if (me)
    {
      me->Fill(status_HE_);
      simpleMap->setBinContent(2,1,status_HE_);
    }
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HO");
  if (me)
    {
      me->Fill(status_HO_);
      simpleMap->setBinContent(3,1,status_HO_);
    }
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HF");
  if (me)
    {
      me->Fill(status_HF_);
      simpleMap->setBinContent(4,1,status_HF_);
    }
  // test for ZDC info
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_ZDC");
  if (me)
    {
      me->Fill(status_ZDC_);
      simpleMap->setBinContent(5,1,status_ZDC_);
    }

  dqmStore_->setCurrentFolder( prefixME_);

 return;
} // void HcalSummaryClient::analyze(void)

void HcalSummaryClient::analyze_subtask(SubTaskSummaryStatus &s)
{
  if (depthME.size()!=4) 
    {
      if (debug_>0) std::cout <<"<HcalSummaryClient::analyze_subtask> Could not get Summary Maps!  # of maps found = "<<depthME.size()<<std::endl;
      return;
    }

  int etabins=0;
  int ieta=-9999;
  int iphi=-9999;
  double bincontent;

  ostringstream name;
  MonitorElement* me;
  TH2F* hist;

  // get overall problem rate histogram for this task
  name.str("");
  name << prefixME_<<"/"<<s.summaryName;
  me=dqmStore_->get(name.str().c_str());
  name.str("");
  double counter=0;
  if (me) // scale histogram to provide failure rate
    {
      hist=me->getTH2F();
      counter=hist->GetBinContent(0,0);
      if (counter>0) 
	{
	  hist->Scale(1./counter);
	  // scale to 0-1 to always maintain consistent coloration
	  hist->SetMaximum(1.);
	  hist->SetMinimum(0.);  // change to some threshold value?
	}
    }

  if (HBpresent_ || HEpresent_ || HFpresent_ || HOpresent_ || ZDCpresent_)
    {
      s.ALLstatus = 0; //number of bad cells
      if (HBpresent_)  s.status[0]=0; // number of bad HB cells
      if (HEpresent_)  s.status[1]=0; // number of bad HE cells;
      if (HOpresent_)  s.status[2]=0; // number of bad HO cells;
      if (HFpresent_)  s.status[3]=0; // number of bad HF cells;
      if (ZDCpresent_) s.status[4]=0; // number of bad ZDC cells;

      for (int d=0;d<4;++d)
	{
	  name.str("");
	  if (d==0) name<< prefixME_<<"/"<<s.problemDir<<"/"<<"HB HE HF Depth 1 "<<s.problemName;
	  else if (d==1) name<< prefixME_<<"/"<<s.problemDir<<"/"<<"HB HE HF Depth 2 "<<s.problemName;
	  else if (d==2) name<< prefixME_<<"/"<<s.problemDir<<"/"<<"HE Depth 3 "<<s.problemName;
	  else if (d==3) name<< prefixME_<<"/"<<s.problemDir<<"/"<<"HO Depth 4 "<<s.problemName;

	  me=dqmStore_->get(name.str().c_str());
	  name.str("");

	  if (!me && debug_>0)  
	    std::cout <<"<HcalSummaryClient::analyze_subtask> CAN'T FIND HISTOGRAM WITH NAME:  "<<name.str().c_str()<<std::endl;
	  else if (me)
	    {
	      hist=me->getTH2F();
	      counter=hist->GetBinContent(0,0); // event counter
	      if (counter>0) // scale by number of events to make rate histogram
		{
		  hist->Scale(1./counter);
		  // scale to 0-1 to always maintain consistent coloration
		  hist->SetMaximum(1.);
		  hist->SetMinimum(0.);  // change to some threshold value?
		}
	      etabins=hist->GetNbinsX();
	      for (int eta=0;eta<etabins;++eta)
		{
		  ieta=CalcIeta(eta,d+1); // skip non-physical bins when counting bad cells
		  if (ieta==-9999) continue;
		  for (int phi=0; phi<72;++phi)
		    {
		      bincontent=hist->GetBinContent(eta+1,phi+1);
		      if (bincontent>s.thresh) // cell is bad if above threshold
			{
			  iphi=phi+1;
			  //cout <<"Found bad cell!\t" <<ieta<<"\t"<<iphi<<"\t"<<d+1<<"\tbincontent = "<<bincontent<<endl;
			  if (isHF(eta,d+1))
			    {
			      if (!HFpresent_) continue;
			      depthME[d]->setBinContent(eta+1,phi+1,1); // Fill bin with a value of 1 if error found
			      /*
			      if (fillUnphysical_)
				{
				  if (abs(ieta)>39 && iphi%4==3)
				    {
				      depthME[d]->setBinContent(eta+1,phi+2,1);
				      depthME[d]->setBinContent(eta+1,phi,1);
				      depthME[d]->setBinContent(eta+1,phi-1,1);
				    }
				  else if (abs(ieta)>20 && iphi%2==1)
				    {
				      depthME[d]->setBinContent(eta+1,phi+2,1);
				    }
				}
			      */
			      // don't count unphysical bins in status
			      if (iphi%2==0) continue;  // skip non-physical phi bins
			      if (abs(ieta)>39 && (iphi%4!=3)) continue;

			      s.status[3]++;
			      s.ALLstatus++;
			    }
			  else if (isHB(eta,d+1))
			    {
			      if (!HBpresent_) continue;
			      depthME[d]->setBinContent(eta+1,phi+1,1);
			      s.status[0]++;
			      s.ALLstatus++;
			    }
			  else if (isHE(eta,d+1))
			    {
			      if (!HEpresent_) continue;
			      depthME[d]->setBinContent(eta+1,phi+1,1);
			      if (abs(ieta)>20 && iphi%2==0) continue; // skip nonphysical bins
			      s.status[1]++;
			      s.ALLstatus++;
			    }
			  else if (isHO(eta,d+1))
			    {
			      if (!HOpresent_) continue;
			      depthME[d]->setBinContent(eta+1,phi+1,1);
			      s.status[2]++;
			      s.ALLstatus++;
			    }
			} // if (bincontent>s.thresh)
		    } // for (int phi=0;...)
		} // for (int eta=0;...)
	    } // if (me)
	} //for (int d=0;d<4;++d)
    } // if (HBpresent_ || HEpresent_ || HFpresent || HOpresent_)
  return;
} //void HcalSummaryClient::analyze_subtask(SubTaskSummaryStatus &s)

void HcalSummaryClient::resetSummaryPlots()
{
  MonitorElement* summary = dqmStore_->get(prefixME_+"/EventInfo/reportSummaryMap");
  // if subdetector present, histogram value set to 0; otherwise, set to -1
  (HBpresent_==true)  ? summary->setBinContent(1,0) : summary->setBinContent(1,-1);
  (HEpresent_==true)  ? summary->setBinContent(2,0) : summary->setBinContent(2,-1);
  (HOpresent_==true)  ? summary->setBinContent(3,0) : summary->setBinContent(3, -1);
  (HFpresent_==true)  ? summary->setBinContent(4,0) : summary->setBinContent(4, -1);
  (ZDCpresent_==true) ? summary->setBinContent(5,0) : summary->setBinContent(5, -1);

  // Set all starting bins to 0, rather than -1
  int ieta=0;
  int iphi=0;

  for (unsigned int d=0;d<depthME.size();++d)
    {
      TH2F* hist=depthME[d]->getTH2F();
      int etabins=hist->GetNbinsX();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<72;++phi)
	    {
	      iphi=phi+1;
	      // Set starting bin value to 0, unless subdetector not present (in which case value is -1)
	      if ((HBpresent_ && isHB(eta,d+1)) ||
		  (HEpresent_ && isHE(eta,d+1)) ||
		  (HOpresent_ && isHO(eta,d+1)) ||
		  (HFpresent_ && isHF(eta,d+1))
		  )
		hist->SetBinContent(eta+1,phi+1,0);
	      else
		hist->SetBinContent(eta+1,phi+1,-1);
	    } // for (int phi=0;...)
	} // for (int eta=0;...)
    } // for (std::vector<MonitorElement*>::iterator depth=depthME.begin();...)

  return;
} // void HcalSummaryClient::resetSummaryPlots()


void HcalSummaryClient::htmlOutput(int& run, time_t& mytime, int& minlumi, int& maxlumi, string& htmlDir, string& htmlName)
{

  if (debug_) std::cout << "Preparing HcalSummaryClient html output ..." << std::endl;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  //htmlFile << "<br>  " << std::endl;
  htmlFile << "<a name=""top""></a>" << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span>" << std::endl;

  std::string startTime=ctime(&mytime);
  htmlFile << "&nbsp;&nbsp;LS:&nbsp;" << std::endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << minlumi << "</span>" << std::endl;
  htmlFile << "-" << std::endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << maxlumi << "</span>" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Start&nbsp;Time:&nbsp;<spa\n style=\"color: rgb(0, 0, 153);\">" << startTime << "</span></h2> " << std::endl;
  
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span> </h2> " << std::endl;


  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;

  /*
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
    htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
    htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  htmlFile << "<br>" << std::endl;
  */

  // Produce the plots to be shown as .png files from existing histograms

  /*
  TH2F* obj2f;
  std::string imgNameMap="";
  std::string imgName;
  gStyle->SetPaintTextFormat("+g");

 
  // Test for now -- let's just dump out global summary histogram
  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");
  obj2f = me->getTH2F();
 

  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  TH2F* simple2f = me->getTH2F();
  */

  // Standard error palette, extended to greys for - values
  static int pcol[40];
  float rgb[20][3];
  
  for( int i=0; i<20; ++i ) 
    {
      //pcol[i]=kGray; // grey -- seems to be red in my version of root?
      pcol[i]=18;
      if ( i < 17 ) 
	{
	  rgb[i][0] = 0.80+0.01*i;
	  rgb[i][1] = 0.00+0.03*i;
	  rgb[i][2] = 0.00;
	} 
      else if ( i < 19 ) 
	{
	  rgb[i][0] = 0.80+0.01*i;
	  rgb[i][1] = 0.00+0.03*i+0.15+0.10*(i-17);
	  rgb[i][2] = 0.00;
	} else if ( i == 19 ) 
	{
	  rgb[i][0] = 0.00;
	  rgb[i][1] = 0.80;
	  rgb[i][2] = 0.00;
	}
      pcol[20+i] = 1101+i; // was 901+i, but root defines colors up to 1000?
      TColor* color = gROOT->GetColor( 1101+i );
      if( ! color ) color = new TColor(1101+i, 0, 0, 0, "" );
      color->SetRGB( rgb[i][0], rgb[i][1], rgb[i][2] );
    } // for (int i=0;i<20;++i)
 
   gStyle->SetPalette(40, pcol);
   gStyle->SetOptStat(0);
   /*
   if( obj2f ) 
     {
       obj2f->SetMinimum(-1.);
       obj2f->SetMaximum(+1.0);
       obj2f->SetOption("colz");
     }

  if (obj2f)// && obj2f->GetEntries()!=0)
    {
      htmlFile << "<table  width=100% border=1><tr>" << std::endl; 
      htmlFile << "<tr align=\"center\">" << std::endl;  
      htmlAnyHisto(run,obj2f,"i#eta","i#phi",92,htmlFile,htmlDir);
      htmlAnyHisto(run,simple2f,"","",92,htmlFile,htmlDir);
      htmlFile <<"</tr></table>"<<std::endl;

    } // if (obj2f)
   */
  
  // Make table that lists all status words for each subdet
  
  htmlFile<<"<hr><br><h2>Summary Values for Each Subdetector</h2><br>"<<std::endl;
  htmlFile << "<table border=\"2\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td>Task</td><td>HB</td><td>HE</td><td>HO</td><td>HF</td><td>ZDC</td><td>HCAL</td></tr>"<<std::endl;
  if (dataFormatMon_.onoff)
    htmlFile<<"<td>Data Format Monitor</td><td>"<<dataFormatMon_.status[0]<<"</td><td>"<<dataFormatMon_.status[1]<<"</td><td>"<<dataFormatMon_.status[2]<<"</td><td>"<<dataFormatMon_.status[3]<<"</td><td>"<<dataFormatMon_.status[4]<<"</td><td>"<<dataFormatMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (digiMon_.onoff)
    htmlFile<<"<td>Digi Monitor</td><td>"<<digiMon_.status[0]<<"</td><td>"<<digiMon_.status[1]<<"</td><td>"<<digiMon_.status[2]<<"</td><td>"<<digiMon_.status[3]<<"</td><td>"<<digiMon_.status[4]<<"</td><td>"<<digiMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (recHitMon_.onoff)
    htmlFile<<"<td>Rec Hit Monitor</td><td>"<<recHitMon_.status[0]<<"</td><td>"<<recHitMon_.status[1]<<"</td><td>"<<recHitMon_.status[2]<<"</td><td>"<<recHitMon_.status[3]<<"</td><td>"<<recHitMon_.status[4]<<"</td><td>"<<recHitMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (pedestalMon_.onoff)
    htmlFile<<"<td>Pedestal Monitor</td><td>"<<pedestalMon_.status[0]<<"</td><td>"<<pedestalMon_.status[1]<<"</td><td>"<<pedestalMon_.status[2]<<"</td><td>"<<pedestalMon_.status[3]<<"</td><td>"<<pedestalMon_.status[4]<<"</td><td>"<<pedestalMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (ledMon_.onoff)
    htmlFile<<"<td>LED Monitor</td><td>"<<ledMon_.status[0]<<"</td><td>"<<ledMon_.status[1]<<"</td><td>"<<ledMon_.status[2]<<"</td><td>"<<ledMon_.status[3]<<"</td><td>"<<ledMon_.status[4]<<"</td><td>"<<ledMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (hotCellMon_.onoff)
    htmlFile<<"<td>Hot Cell Monitor</td><td>"<<hotCellMon_.status[0]<<"</td><td>"<<hotCellMon_.status[1]<<"</td><td>"<<hotCellMon_.status[2]<<"</td><td>"<<hotCellMon_.status[3]<<"</td><td>"<<hotCellMon_.status[4]<<"</td><td>"<<hotCellMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (deadCellMon_.onoff)
    htmlFile<<"<td>Dead Cell Monitor</td><td>"<<deadCellMon_.status[0]<<"</td><td>"<<deadCellMon_.status[1]<<"</td><td>"<<deadCellMon_.status[2]<<"</td><td>"<<deadCellMon_.status[3]<<"</td><td>"<<deadCellMon_.status[4]<<"</td><td>"<<deadCellMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (trigPrimMon_.onoff)
    htmlFile<<"<td>Trigger Primitive Monitor</td><td>"<<trigPrimMon_.status[0]<<"</td><td>"<<trigPrimMon_.status[1]<<"</td><td>"<<trigPrimMon_.status[2]<<"</td><td>"<<trigPrimMon_.status[3]<<"</td><td>"<<trigPrimMon_.status[4]<<"</td><td>"<<trigPrimMon_.ALLstatus<<"</td></tr>"<<std::endl;
  if (caloTowerMon_.onoff)
    htmlFile<<"<td>CaloTower Monitor</td><td>"<<caloTowerMon_.status[0]<<"</td><td>"<<caloTowerMon_.status[1]<<"</td><td>"<<caloTowerMon_.status[2]<<"</td><td>"<<caloTowerMon_.status[3]<<"</td><td>"<<caloTowerMon_.status[4]<<"</td><td>"<<caloTowerMon_.ALLstatus<<"</td></tr>"<<std::endl;

  htmlFile<<"<td><font color = \"blue\">Overall Status</font></td>"<<"<td><font color = \"blue\">"<<status_HB_<<"</font></td><td><font color = \"blue\">"<<status_HE_<<"</font></td><td><font color = \"blue\">"<<status_HO_<<"</font></td><td><font color = \"blue\">"<<status_HF_<<"</font></td><td><font color = \"blue\">"<< status_ZDC_<<"</font></td><td><font color = \"blue\">"<<   status_global_<<"</font></td></tr>"<<std::endl;
  htmlFile <<"</tr></table>"<<std::endl;
 
  htmlFile <<"<br><h2> A note on Status Values </h2><br>"<<std::endl;
  htmlFile <<"Status values in each subdetector task represent the average fraction of good channels per event.  (For example, a value of .99 in the HB Hot Cell monitor means that, on average, 1% of the cells in HB are hot.)  Status values should range from 0 to 1, with a perfectly-functioning detector will have all status values = 1.  If the status is unknown, a value of -1 or \"--\" will be shown. <br>"<<std::endl;
  htmlFile <<"<br>The HCAL status values for each task are a weighted average from each subdetector.  Weights are assigned as (# of cells in the subdetector)/(total # of cells being checked).<br>"<<std::endl;
  htmlFile <<"<br>The overall Status Values at the bottom of the table are a combination of the individual status values.  These values are not quite the same as the overall fraction of good channels in an event, because of ambiguities in the eta-phi plots.  (The summary code does not store the results of monitor tests on individual events, and thus can't tell the difference between a run where the digi monitor failed in the first half of events and the dead cell monitor failed the second half and a run in which the digi and dead cell monitors were both bad only for the first 50% of the run.  For the moment, the errors from the different monitors are added together, but this can lead to double-counting, and an overall status value less than the individual values.)"<<std::endl;

  htmlFile <<"<br><hr><br>"<<std::endl;
  htmlFile <<"Run #: "<<run<<"&nbsp;&nbsp;&nbsp;&nbsp Starting Time: "<<startTime<<"&nbsp;&nbsp;&nbsp;&nbsp;";
  htmlFile<<"Luminosity blocks: "<<minlumi<<" - "<<maxlumi <<"&nbsp;&nbsp;&nbsp;&nbsp";
  htmlFile <<"# of events: "<<ievt_<<"&nbsp;&nbsp;&nbsp;&nbsp";
  if (dataFormatMon_.onoff)
    htmlFile <<"  Data Format Status:  HB: "<<dataFormatMon_.status[0]<<"  HE: "<<dataFormatMon_.status[1]<<"  HO: "<<dataFormatMon_.status[2]<<"  HF: "<<dataFormatMon_.status[3]<<"  ZDC: "<<dataFormatMon_.status[4]<<std::endl;
  if (digiMon_.onoff)
    htmlFile <<"  Digi Monitor Status:  HB: "<<digiMon_.status[0]<<"  HE: "<<digiMon_.status[1]<<"  HO: "<<digiMon_.status[2]<<"  HF: "<<digiMon_.status[3]<<"  ZDC: "<<digiMon_.status[4]<<std::endl;
  if (recHitMon_.onoff)
    htmlFile <<"  RecHit Monitor Status:  HB: "<<recHitMon_.status[0]<<"  HE: "<<recHitMon_.status[1]<<"  HO: "<<recHitMon_.status[2]<<"  HF: "<<recHitMon_.status[3]<<"  ZDC: "<<recHitMon_.status[4]<<std::endl;
  if (pedestalMon_.onoff)
    htmlFile <<"  Pedestal Monitor Status:  HB: "<<pedestalMon_.status[0]<<"  HE: "<<pedestalMon_.status[1]<<"  HO: "<<pedestalMon_.status[2]<<"  HF: "<<pedestalMon_.status[3]<<"  ZDC: "<<pedestalMon_.status[4]<<std::endl;
  if (ledMon_.onoff)
    htmlFile <<"  LED Monitor Status:  HB: "<<ledMon_.status[0]<<"  HE: "<<ledMon_.status[1]<<"  HO: "<<ledMon_.status[2]<<"  HF: "<<ledMon_.status[3]<<"  ZDC: "<<ledMon_.status[4]<<std::endl;
  if (hotCellMon_.onoff)
    htmlFile <<"  Hot Cell Monitor Status:  HB: "<<hotCellMon_.status[0]<<"  HE: "<<hotCellMon_.status[1]<<"  HO: "<<hotCellMon_.status[2]<<"  HF: "<<hotCellMon_.status[3]<<"  ZDC: "<<hotCellMon_.status[4]<<std::endl;
  if (deadCellMon_.onoff)
    htmlFile <<"  Dead Cell Monitor Status:  HB: "<<deadCellMon_.status[0]<<"  HE: "<<deadCellMon_.status[1]<<"  HO: "<<deadCellMon_.status[2]<<"  HF: "<<deadCellMon_.status[3]<<"  ZDC: "<<deadCellMon_.status[4]<<std::endl;
  if (trigPrimMon_.onoff)
    htmlFile <<"  Trigger Primitive Monitor Status:  HB: "<<trigPrimMon_.status[0]<<"  HE: "<<trigPrimMon_.status[1]<<"  HO: "<<trigPrimMon_.status[2]<<"  HF: "<<trigPrimMon_.status[3]<<"  ZDC: "<<trigPrimMon_.status[4]<<std::endl;
  if (caloTowerMon_.onoff)
    htmlFile <<"  CaloTower Monitor Status:  HB: "<<caloTowerMon_.status[0]<<"  HE: "<<caloTowerMon_.status[1]<<"  HO: "<<caloTowerMon_.status[2]<<"  HF: "<<caloTowerMon_.status[3]<<"  ZDC: "<<caloTowerMon_.status[4]<<std::endl;
  htmlFile <<"  OVERALL STATUS:  "<<status_global_<<std::endl;
  htmlFile.close();
} // void htmlOutput(...)


bool HcalSummaryClient::hasErrors_Temp()
{
  float error=0.8;
  if (status_HB_<error && status_HB_>-1) return true;
  if (status_HE_<error && status_HE_>-1) return true;
  if (status_HO_<error && status_HO_>-1) return true;
  if (status_HF_<error && status_HF_>-1) return true;
  if (status_ZDC_<error && status_ZDC_>-1) return true;
  return false;
} // bool HcalSummaryClient::hasErrors_Temp()

bool HcalSummaryClient::hasWarnings_Temp()
{
  float error=0.95;
  if (status_HB_<error && status_HB_>-1) return true;
  if (status_HE_<error && status_HE_>-1) return true;
  if (status_HO_<error && status_HO_>-1) return true;
  if (status_HF_<error && status_HF_>-1) return true;
  if (status_ZDC_<error && status_ZDC_>-1) return true;
  return false;
} // bool HcalSummaryClient::hasWarnings_Temp()
