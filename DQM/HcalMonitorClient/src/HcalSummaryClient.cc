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
  // At the moment, only hot/dead/pedestal comply with correct format of histograms; ignore all others
  dataFormatMon_.onoff=(ps.getUntrackedParameter<bool>("DataFormatClient",false));
  digiMon_.onoff=(ps.getUntrackedParameter<bool>("DigiClient",false));
  recHitMon_.onoff=(ps.getUntrackedParameter<bool>("RecHitClient",false));
  pedestalMon_.onoff=(ps.getUntrackedParameter<bool>("PedestalClient",false));
  pedestalMon_.onoff=false; // don't include pedestal monitoring in overall data quality?
  //ledMon_.onoff=(ps.getUntrackedParameter<bool>("LEDClient",false));
  hotCellMon_.onoff=(ps.getUntrackedParameter<bool>("HotCellClient",false));
  deadCellMon_.onoff=(ps.getUntrackedParameter<bool>("DeadCellClient",false));
  //trigPrimMon_.onoff=(ps.getUntrackedParameter<bool>("TrigPrimClient",false));
  //caloTowerMon_.onoff=(ps.getUntrackedParameter<bool>("CaloTowerClient",false));

  // Set histogram problem names & directories  for each subtask
  dataFormatMon_.baseProblemName  = "DataFormatMonitor/ HardwareWatchCells"; // untested
  digiMon_.baseProblemName        = "DigiMonitor_Hcal/ ProblemDigis";
  recHitMon_.baseProblemName      = "RecHitMonitor_Hcal/ ProblemRecHits";
  pedestalMon_.baseProblemName    = "BaselineMonitor_Hcal/ ProblemPedestals";
  ledMon_.baseProblemName         = "";
  hotCellMon_.baseProblemName     = "HotCellMonitor_Hcal/ ProblemHotCells";
  deadCellMon_.baseProblemName    = "DeadCellMonitor_Hcal/ ProblemDeadCells";
  trigPrimMon_.baseProblemName    = "";
  caloTowerMon_.baseProblemName   = "";

  dataFormatMon_.problemName  = " Hardware Watch Cells";
  digiMon_.problemName        = " Problem Digi Rate";
  recHitMon_.problemName      = " Problem RecHit Rate";
  pedestalMon_.problemName    = " Problem Pedestal Rate";
  ledMon_.problemName         = "";
  hotCellMon_.problemName     = " Problem Hot Cell Rate";
  deadCellMon_.problemName    = " Problem Dead Cell Rate";
  trigPrimMon_.problemName    = "";
  caloTowerMon_.problemName   = "";

  dataFormatMon_.problemDir   = "DataFormatMonitor";
  digiMon_.problemDir         = "DigiMonitor_Hcal/problem_digis";
  recHitMon_.problemDir       = "RecHitMonitor_Hcal/problem_rechits";
  pedestalMon_.problemDir     = "BaselineMonitor_Hcal/problem_pedestals";
  ledMon_.problemDir          = "";
  hotCellMon_.problemDir      = "HotCellMonitor_Hcal/problem_hotcells";
  deadCellMon_.problemDir     = "DeadCellMonitor_Hcal/problem_deadcells";
  trigPrimMon_.problemDir     = "";
  caloTowerMon_.problemDir    = "";
  
  dataFormatMon_.ievtName   = "DataFormatMonitor/Data Format Task Event Number";
  digiMon_.ievtName         = "DigiMonitor_Hcal/Digi Task Event Number";
  recHitMon_.ievtName       = "RecHitMonitor_Hcal/RecHit Task Event Number";
  pedestalMon_.ievtName     = "BaselineMonitor_Hcal/Pedestal Task Event Number";
  ledMon_.ievtName          = "";
  hotCellMon_.ievtName      = "HotCellMonitor_Hcal/Hot Cell Task Event Number";
  deadCellMon_.ievtName     = "DeadCellMonitor_Hcal/Dead Cell Task Event Number";
  trigPrimMon_.ievtName     = "";
  caloTowerMon_.ievtName    = "";


  // All initial status floats set to -1 (unknown)
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

  // Assume subdetectors absent at start
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;
  ZDCpresent_=0;

  // Set eta, phi boundaries for overall report summary map
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
      dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );  histo<<"Hcal_"<<subdets[i].c_str();
      me=dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
      if (me)
	dqmStore_->removeElement(me->getName());
      me = dqmStore_->bookFloat(histo.str().c_str());
      me->Fill(-1); // set status to unknown at startup
      histo.str("");
    }

  // Make overall 2D histogram
  dqmStore_->setCurrentFolder(prefixME_+"/EventInfo/");
  histo<<"advancedReportSummaryMap";
  me=dqmStore_->get(prefixME_+"/EventInfo/"+histo.str().c_str());
  if (me)
    dqmStore_->removeElement(me->getName());
  me = dqmStore_->book2D(histo.str().c_str(), histo.str().c_str(), 
			 85,-42.5,42.5,
			 72,0.5,72.5);
  // Set histogram values to -1
  // Set all bins to "unknown" to start
  for (int ieta=1;ieta<=etaBins_;++ieta)
    for (int iphi=1; iphi<=phiBins_;++iphi)
      me->setBinContent(ieta,iphi,-1);
  
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

  MonitorElement* advancedMap = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");
  // Set all bins to "unknown" to start

  if (advancedMap)
    {
      for (int ieta=1;ieta<=etaBins_;++ieta)
	for (int iphi=1; iphi<=phiBins_;++iphi)
	  advancedMap->setBinContent(ieta,iphi,-1);
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

  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");
  // Set all bins to "unknown" to start
  for (int ieta=1;ieta<=etaBins_;++ieta)
    for (int iphi=1; iphi<=phiBins_;++iphi)
      reportMap->setBinContent(ieta,iphi,-1);


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

 if (HBpresent_) status_HB_=0;
 if (HEpresent_) status_HE_=0;
 if (HOpresent_) status_HO_=0;
 if (HFpresent_) status_HF_=0;
 if (ZDCpresent_) status_ZDC_=0;
 if (HBpresent_ || HEpresent_ || HOpresent_ || HFpresent_ ) // don't include ZDC yet
   status_global_=0;

 // Set starting histogram values to 0
 if (HBpresent_) resetSummaryPlot(1); 
 if (HEpresent_) resetSummaryPlot(2);
 if (HOpresent_) resetSummaryPlot(3);
 if (HFpresent_) resetSummaryPlot(4);

 // Calculate status values for individual tasks
 if (dataFormatMon_.IsOn()) New_analyze_subtask(dataFormatMon_);
 if (digiMon_.IsOn()) New_analyze_subtask(digiMon_);
 if (recHitMon_.IsOn()) New_analyze_subtask(recHitMon_);
 if (pedestalMon_.IsOn()) New_analyze_subtask(pedestalMon_);
 if (ledMon_.IsOn()) analyze_subtask(ledMon_);
 if (hotCellMon_.IsOn()) New_analyze_subtask(hotCellMon_);
 if (deadCellMon_.IsOn()) New_analyze_subtask(deadCellMon_);
 if (trigPrimMon_.IsOn()) analyze_subtask(trigPrimMon_);
 if (caloTowerMon_.IsOn()) analyze_subtask(caloTowerMon_);

 // Okay, we've got the individual tasks; now form the combined value

  int totalcells=0;
  std::map<std::string, int>::const_iterator it;
  if (HBpresent_)
    {
      status_global_+=status_HB_;
      it=subdetCells_.find("HB");
      totalcells+=it->second;
      status_HB_/=it->second;
      status_HB_=max(0.,1-status_HB_); // converts fraction of bad channels to good fraction
    }
  if (HEpresent_)
    {
      status_global_+=status_HE_;
      it=subdetCells_.find("HE");
      totalcells+=it->second;
      status_HE_/=it->second;
      status_HE_=max(0.,1-status_HE_); // converts fraction of bad channels to good fraction
    }
  if (HOpresent_)
    {
      status_global_+=status_HO_;
      it=subdetCells_.find("HO");
      totalcells+=it->second;
      status_HO_/=it->second;
      status_HO_=max(0.,1-status_HO_); // converts fraction of bad channels to good fraction
    }
  if (HFpresent_)
    {
      status_global_+=status_HF_;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      status_HF_/=it->second;
      status_HF_=max(0.,1-status_HF_); // converts fraction of bad channels to good fraction
    }
  /*
 if (ZDCpresent_)
    {
    status_global_+=status_ZDC_;
      it=subdetCells_.find("ZDC");
      totalcells+=it->second;
      status_ZDC_/=it->second;
      status_ZDC_=max(0.,1-status_ZDC_); // converts fraction of bad channels to good fraction

    }
  */
  
  if (totalcells==0)
    status_global_=-1;
  else
    {
      status_global_/=totalcells;
      status_global_=max(0.,1-status_global_); // convert to good fraction
      // Now loop over cells in reportsummarymap, changing from bad fraction to good

      int eta,phi;
      for (int ieta=1;ieta<=etaBins_;++ieta)
	{
	  eta=ieta+int(etaMin_)-1;
	  for (int iphi=1; iphi<=phiBins_;++iphi)
	    {
	      if (reportMap->getBinContent(ieta,iphi)>-1)
		{
		  phi=iphi+int(phiMin_)-1;
		  if (abs(eta)>20 && phi%2!=1)
		    continue;
		  if (abs(eta)>39 &&phi%4!=3)
		    continue;
		  reportMap->setBinContent(ieta,iphi,max(0.,(double)(1-reportMap->getBinContent(ieta,iphi))));

		  if (fillUnphysical_)
		    {
		      // fill even phi cells in region where cells span 10 degrees in phi
		      // ("True" cell values are phi=1,3,5,7,...) 
		      if (abs(eta)>20 && abs(eta)<40 && phi%2==1 &&phi<73)
			{
			  reportMap->setBinContent(ieta,iphi+1,reportMap->getBinContent(ieta,iphi));
			}
		      
		      // fill all cells in region where cells span 20 degrees in phi
		      // (actual cell phi values are 3,7,11,...)
		      if (abs(eta)>39 && phi%4==3 && phi<73)
			{
			  reportMap->setBinContent(ieta,iphi+1,reportMap->getBinContent(ieta,iphi));
			  reportMap->setBinContent(ieta,iphi-1,reportMap->getBinContent(ieta,iphi));
			  reportMap->setBinContent(ieta,iphi-2,reportMap->getBinContent(ieta,iphi));
			}
		    }
		} //if (bincontent>-1)
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // else (totalcells>0)

  // Now set the status words
  MonitorElement* me;
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );
  
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
  double HBstatus=0;
  double HEstatus=0;
  double HOstatus=0;
  double HFstatus=0;
  double ZDCstatus=-1; // not yet implemented
  double ALLstatus=0;


  double etamin, etamax, phimin, phimax;
  int etabins, phibins;
  int eta, phi;
  double bincontent;

  //cout <<"Running Summary Client"<<endl;

  ostringstream name;
  MonitorElement* me;
  TH2F* hist;
  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");

  // No longer use ievtTask for normalization; 
  // use the underflow bin (0,0) of the 2D histograms to get # of events,
  // just as with renderPlugins
  /*
  int ievtTask=-1;
  name.str("");
  name <<prefixME_<<"/"<<s.ievtName;
  me = dbe_->get(name.str().c_str());
  name.str("");
  if (me)
    {
      string s = me->valueString();
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievtTask);
      if ( debug_>0 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
    }
  */
  // Scale overall problem plot
  name.str("");
  name << prefixME_<<"/"<<s.baseProblemName;
  me=dqmStore_->get(name.str().c_str());

  double counter=0;
  if (me)
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


  // Layer 1 HB& HF
  if (HBpresent_ || HFpresent_)
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HB HF Depth 1 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      name.str("");
      if (!me && debug_>0)  std::cout <<"<HcalSummaryClient::analyze_subtask> CAN'T FIND HISTOGRAM WITH NAME:  "<<name.str().c_str()<<std::endl;
      if (me)
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
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      eta=ieta+int(etamin)-1;
		      phi=iphi+int(phimin)-1;
		      reportMap->Fill(eta,phi,bincontent);
		      if (abs(eta)<17 && HBpresent_) // HB
			{
			  HBstatus+=bincontent;
			}
		      else if (HFpresent_)
			{
			  if (phi%2==0) continue; // skip non-physical phi bins
			  if (abs(eta)>39 && phi%4!=3) continue; // skip non-physical phi bins
			  HFstatus+=bincontent;
			}
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
    } // if (HBpresent_ || HFpresent)

  // Layer 2 HB& HF
  if (HBpresent_ || HFpresent_)
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HB HF Depth 2 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }

	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      eta=ieta+int(etamin)-1;
		      phi=iphi+int(phimin)-1;
		      reportMap->Fill(eta,phi,bincontent);
		      if (abs(eta)<17 && HBpresent_) // HB
			{
			  HBstatus+=bincontent;
			}
		      else if (HFpresent_)
			{
			  if (phi%2==0) continue; // skip non-physical phi bins
			  if (abs(eta)>39 && phi%4!=3) continue; // skip non-physical phi bins
			  HFstatus+=bincontent;
			}
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
    } // if (HBpresent_ || HFpresent)

  // Layer 4 HO & ZDC
  if (HOpresent_ || ZDCpresent_)
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HO ZDC "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      eta=ieta+int(etamin)-1;
		      phi=iphi+int(phimin)-1;
		      reportMap->Fill(eta,phi,bincontent);
		      if (abs(eta)<42 && HOpresent_) // HO
			{
			  HOstatus+=bincontent;
			}
		      else if (ZDCpresent_)
			{
			  ZDCstatus+=bincontent;
			}
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
    } // if (HOpresent_ || ZDCpresent)

  // Layer 1 HE
  if (HEpresent_) 
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HE Depth 1 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      //cout <<"Got "<<name.str().c_str()<<endl;
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  eta=ieta+int(etamin)-1;
		  phi=iphi+int(phimin)-1;
		  if (abs(eta)>20 && phi%2==0) continue; // skip non-physical phi bins

		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      reportMap->Fill(eta,phi,bincontent);
		      HEstatus+=bincontent;
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
      //cout <<"HEstatus (depth1)= "<<HEstatus<<endl;
    } // if (HEpresent_)
  
   // Layer 2 HE
  if (HEpresent_) 
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HE Depth 2 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      //cout <<"Got "<<name.str().c_str()<<endl;
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  eta=ieta+int(etamin)-1;
		  phi=iphi+int(phimin)-1;
		  if (abs(eta)>20 && phi%2==0) continue; // skip non-physical phi bins
		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      eta=ieta+int(etamin)-1;
		      reportMap->Fill(eta,phi,bincontent); 
		      HEstatus+=bincontent;
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
      //cout <<"HEstatus (depth2)= "<<HEstatus<<endl;
    } // if (HEpresent_)

  // HE Depth 3
  if (HEpresent_) 
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HE Depth 3 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      //cout <<"Got "<<name.str().c_str()<<endl;
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  etamin=hist->GetXaxis()->GetXmin();
	  etamax=hist->GetXaxis()->GetXmax();
	  phimin=hist->GetYaxis()->GetXmin();
	  phimax=hist->GetYaxis()->GetXmax();
	  for (int ieta=1;ieta<=etabins;++ieta)
	    {
	      for (int iphi=1; iphi<=phibins;++iphi)
		{
		  
		  eta=ieta+int(etamin)-1;
		  phi=iphi+int(phimin)-1;
		  if (abs(eta)>20 && phi%2==0) continue; // skip non-physical phi bins
		  bincontent=hist->GetBinContent(ieta,iphi);
		  if (bincontent>0)
		    {
		      reportMap->Fill(eta,phi,bincontent);
		      HEstatus+=bincontent;
		    } // if (bincontent>0)
		} // for (int iphi=1;...)
	    } // for (int ieta=1;...)
	} // if (me)
      //cout <<"HEstatus (depth3)= "<<HEstatus<<endl;
    } // if (HEpresent_)

  ALLstatus=HBstatus+HEstatus+HOstatus+HFstatus;
  int totalcells=0;
  std::map<std::string, int>::const_iterator it;
  if (HBpresent_)
    {
      status_HB_+=HBstatus;
      it=subdetCells_.find("HB");
      totalcells+=it->second;
      HBstatus/=it->second;
      HBstatus=1-HBstatus; // converts fraction of bad channels to good fraction
      if (HBstatus<0) HBstatus=0;
      s.status[0]=HBstatus;
    }
  if (HEpresent_)
    {
      status_HE_+=HEstatus;
      it=subdetCells_.find("HE");
      totalcells+=it->second;
      HEstatus/=it->second;
      HEstatus=1-HEstatus; // converts fraction of bad channels to good fraction
      if (HEstatus<0)
	HEstatus=0;
      s.status[1]=HEstatus;
    }

  if (HOpresent_)
    {
      status_HO_+=HOstatus;
      it=subdetCells_.find("HO");
      totalcells+=it->second;
      HOstatus/=it->second;
      HOstatus=1-HOstatus; // converts fraction of bad channels to good fraction
      if (HOstatus<0)
	HOstatus=0;
      s.status[2]=HOstatus;
    }
  if (HFpresent_)
    {
      status_HF_+=HFstatus;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      HFstatus/=it->second;
      HFstatus=1-HFstatus; // converts fraction of bad channels to good fraction
      if (HFstatus<0)
	HFstatus=0;
      s.status[3]=HFstatus;
    }
  /*
 if (ZDCpresent_)
    {
      status_ZDC_+=ZDCstatus;
      it=subdetCells_.find("ZDC");
      totalcells+=it->second;
      ZDCstatus/=it->second;
      ZDCstatus=1-ZDCstatus; // converts fraction of bad channels to good fraction
      if (ZDCstatus<0)
        ZDCstatus=0;
      s.status[4]=ZDCStatus;
    }
  */
  if (totalcells>0)
    {
      ALLstatus/=totalcells;
      ALLstatus=1-ALLstatus;
      if (ALLstatus<0)
	ALLstatus=0;
      s.ALLstatus=ALLstatus;
    }

  if (debug_>0)
    {
      std::cout <<s.problemDir<<std::endl;
      std::cout <<"ReportSummary:  HB = "<<HBstatus<<std::endl;
      std::cout <<"ReportSummary: HE = "<<HEstatus<<std::endl;
      std::cout <<"ReportSummary: HO = "<<HOstatus<<std::endl;
      std::cout <<"ReportSummary: HF = "<<HFstatus<<std::endl;
      std::cout <<"ReportSummary: TOTAL = "<<s.ALLstatus<<std::endl;
      std::cout <<"Avg # of bad cells: "<<std::endl;
      std::cout <<"sumHB = "<<status_HB_<<std::endl;
      std::cout <<"sumHE = "<<status_HE_<<std::endl;
      std::cout <<"sumHO = "<<status_HO_<<std::endl;
      std::cout <<"sumHF = "<<status_HF_<<std::endl;
      std::cout <<"________________"<<std::endl;
      
    }

  return;
} //void HcalSummaryClient::analyze_subtask(SubTaskSummaryStatus &s)


void HcalSummaryClient::New_analyze_subtask(SubTaskSummaryStatus &s)
{
  double HBstatus=0;
  double HEstatus=0;
  double HOstatus=0;
  double HFstatus=0;
  //double ZDCstatus=-1; // not yet implemented
  double ALLstatus=0;

  int etabins, phibins;
  int ieta=-9999;
  int iphi=-9999;
  double bincontent;

  ostringstream name;
  MonitorElement* me;
  TH2F* hist;
  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");

  // No longer use ievtTask for normalization; 
  // use the underflow bin (0,0) of the 2D histograms to get # of events,
  // just as with renderPlugins

  // Scale overall problem plot
  name.str("");
  name << prefixME_<<"/"<<s.baseProblemName;
  me=dqmStore_->get(name.str().c_str());

  double counter=0;
  if (me)
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

  // Layers 1&2  HB HE HF
  if (HBpresent_ || HEpresent_ || HFpresent_)
    {
      for (int d=1;d<=2;++d)
	{
	  int HF_eta_offset=72;
	  if (d==2) HF_eta_offset=42;
	  name.str("");
	  name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HB HE HF Depth "<<d<<" "<<s.problemName;
	  me=dqmStore_->get(name.str().c_str());
	  name.str("");
	  if (!me && debug_>0)  std::cout <<"<HcalSummaryClient::analyze_subtask> CAN'T FIND HISTOGRAM WITH NAME:  "<<name.str().c_str()<<std::endl;
	  if (me)
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
	      etabins=hist->GetNbinsX();
	      phibins=hist->GetNbinsY();
	      
	      for (int eta=1;eta<=etabins;++eta)
		{
		  for (int phi=1; phi<=phibins;++phi)
		    {
		      bincontent=hist->GetBinContent(eta,phi);
		      if (bincontent>0)
			{
			  ieta=CalcIeta(eta,d);
			  iphi=phi;
			  if (eta<14 || eta>HF_eta_offset)
			    {
			      if (eta<14) ieta--;
			      else        ieta++;
			      if (!HFpresent_) continue;
			      if (iphi%2==0) continue; //skip non-physical phi bins
			      if ((eta<3 || eta>83) && phi%4!=3) continue; // skip non-physical phi bins
			      HFstatus+=bincontent;
			    }
			  else if (abs(ieta)<17)
			    {
			      if (!HBpresent_) continue;
			      HBstatus+=bincontent;
			    }
			  else if (ieta!=-9999)
			    {
			      if (!HEpresent_) continue;
			      if (abs(ieta)>20 && iphi%2==0) continue;
			      HEstatus+=bincontent;
			    }

			  reportMap->Fill(ieta,iphi,bincontent); // need to adjust reportMap to contain shifted ieta bins

			} // if (bincontent>0)
		    } // for (int iphi=1;...)
		} // for (int ieta=1;...)
	    } // if (me)
	} //for (int d=1;d<=2;++d)
    } // if (HBpresent_ || HEpresent_ || HFpresent)

  // Layer 3 HE
  if (HEpresent_)
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HE Depth 3 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();
	  
	  for (int eta=1;eta<=etabins;++eta)
	    {
	      for (int phi=1; phi<=phibins;++phi)
		{
		  ieta=binmapd3[eta-1];
		  if (ieta==-9999) continue;
		  iphi=phi;
		  if (abs(ieta)>16 && iphi%2==0) continue; // skip unphysical cells
		  bincontent=hist->GetBinContent(eta,phi);
		  if (bincontent>0)
		    {
		      reportMap->Fill(ieta,iphi,bincontent);
		      HEstatus+=bincontent;
		    } // if (bincontent>0)
		}
	    }
	}
    } // if (HEpresent_)

  // Layer 4 HO 
  if (HOpresent_)
    {
      name.str("");
      name <<prefixME_<<"/"<<s.problemDir<<"/"<<"HO Depth 4 "<<s.problemName;
      me=dqmStore_->get(name.str().c_str());
      
      if (me)
	{
	  hist=me->getTH2F();
	  if (counter>0) 
	    {
	      hist->Scale(1./counter);
	      // scale to 0-1 to always maintain consistent coloration
	      hist->SetMaximum(1.);
	      hist->SetMinimum(0.);  // change to some threshold value?
	    }
	  etabins=hist->GetNbinsX();
	  phibins=hist->GetNbinsY();

	  for (int eta=1;eta<=etabins;++eta)
	    {
	      for (int phi=1; phi<=phibins;++phi)
		{
		  bincontent=hist->GetBinContent(eta,phi);
		  if (bincontent>0)
		    {
		      ieta=eta-16;
		      iphi=phi;
		      reportMap->Fill(ieta,iphi,bincontent);
		      HOstatus+=bincontent;
		    } // if (bincontent>0)
		} // for (int phi=1;...)
	    } // for (int eta=1;...)
	} // if (me)
    } // if (HOpresent_ )

  ALLstatus=HBstatus+HEstatus+HOstatus+HFstatus;
  int totalcells=0;
  std::map<std::string, int>::const_iterator it;
  if (HBpresent_)
    {
      status_HB_+=HBstatus;
      it=subdetCells_.find("HB");
      totalcells+=it->second;
      HBstatus/=it->second;
      HBstatus=1-HBstatus; // converts fraction of bad channels to good fraction
      if (HBstatus<0) HBstatus=0;
      s.status[0]=HBstatus;
    }
  if (HEpresent_)
    {
      status_HE_+=HEstatus;
      it=subdetCells_.find("HE");
      totalcells+=it->second;
      HEstatus/=it->second;
      HEstatus=1-HEstatus; // converts fraction of bad channels to good fraction
      if (HEstatus<0)
	HEstatus=0;
      s.status[1]=HEstatus;
    }

  if (HOpresent_)
    {
      status_HO_+=HOstatus;
      it=subdetCells_.find("HO");
      totalcells+=it->second;
      HOstatus/=it->second;
      HOstatus=1-HOstatus; // converts fraction of bad channels to good fraction
      if (HOstatus<0)
	HOstatus=0;
      s.status[2]=HOstatus;
    }
  if (HFpresent_)
    {
      status_HF_+=HFstatus;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      HFstatus/=it->second;
      HFstatus=1-HFstatus; // converts fraction of bad channels to good fraction
      if (HFstatus<0)
	HFstatus=0;
      s.status[3]=HFstatus;
    }

  if (totalcells>0)
    {
      ALLstatus/=totalcells;
      ALLstatus=1-ALLstatus;
      if (ALLstatus<0)
	ALLstatus=0;
      s.ALLstatus=ALLstatus;
    }

  if (debug_>0)
    {
      std::cout <<s.problemDir<<std::endl;
      std::cout <<"ReportSummary:  HB = "<<HBstatus<<std::endl;
      std::cout <<"ReportSummary: HE = "<<HEstatus<<std::endl;
      std::cout <<"ReportSummary: HO = "<<HOstatus<<std::endl;
      std::cout <<"ReportSummary: HF = "<<HFstatus<<std::endl;
      std::cout <<"ReportSummary: TOTAL = "<<s.ALLstatus<<std::endl;
      std::cout <<"Avg # of bad cells: "<<std::endl;
      std::cout <<"sumHB = "<<status_HB_<<std::endl;
      std::cout <<"sumHE = "<<status_HE_<<std::endl;
      std::cout <<"sumHO = "<<status_HO_<<std::endl;
      std::cout <<"sumHF = "<<status_HF_<<std::endl;
      std::cout <<"________________"<<std::endl;
      
    }
  return;
} //void HcalSummaryClient::New_analyze_subtask(SubTaskSummaryStatus &s)






void HcalSummaryClient::resetSummaryPlot(int Subdet)
{
  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");
  if (!reportMap)
    {
      std::cout <<"<HcalSummaryClient::resetSummaryPlot> Could not get advancedReportSummaryMap!"<<std::endl;
      return;
    }
  TH2F* hist=reportMap->getTH2F();
  int etabins=hist->GetNbinsX();
  int phibins=hist->GetNbinsY();
  double etamin=hist->GetXaxis()->GetXmin();
  double phimin=hist->GetYaxis()->GetXmin();
  int eta,phi;

  //  Loop over all bins for problem report; set their contents to 0 if 
  // they have valid cell ID
  for (int ieta=1;ieta<=etabins;++ieta)
    {
      for (int iphi=1; iphi<=phibins;++iphi)
	{
	  for (int d=1;d<=4;++d)
	    {
	      eta=ieta+int(etamin)-1;
	      phi=iphi+int(phimin)-1;
	      if (validDetId((HcalSubdetector)Subdet,eta,phi,d))
		hist->SetBinContent(ieta,iphi,0.);
	    }

	} // for (int iphi=1;iphi<=phibins;++iphi)
    } // for (int ieta=1; ieta<=etabins;++ieta)
  return;
} // void HcalSummaryClient::resetSummaryPlot(int Subdet)


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

  TH2F* obj2f;
  std::string imgNameMap="";
  std::string imgName;
  gStyle->SetPaintTextFormat("+g");

  // Test for now -- let's just dump out global summary histogram
  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/advancedReportSummaryMap");
  obj2f = me->getTH2F();

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  TH2F* simple2f = me->getTH2F();

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
