#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>

#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"

#include "DQMServices/Core/interface/DQMStore.h"


#include "DQM/HcalMonitorClient/interface/HcalMonitorClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDataFormatClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDigiClient.h"
#include "DQM/HcalMonitorClient/interface/HcalRecHitClient.h"
#include "DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h"
#include "DQM/HcalMonitorClient/interface/HcalPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalHotCellClient.h"

#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"


using namespace cms;
using namespace edm;
using namespace std;


HcalSummaryClient::HcalSummaryClient(const ParameterSet& ps)
{
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "Hcal");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // eta, phi boundaries -- need to put them in client as well as monitor?
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 42.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -42.5);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);

  for(int i=0; i<4; i++) subDetsOn_[i] = false;


  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB")
      subDetsOn_[0]=true;
    else if(subdets[i]=="HE") 
      subDetsOn_[1]=true;
    else if(subdets[i]=="HO") 
      subDetsOn_[2]=true;
    else if(subdets[i]=="HF")
      subDetsOn_[3]=true;

  } // for (unsigned int i=0; i<subdets.size();i++)

  // Check subtasks
  dataFormatClient_ = ps.getUntrackedParameter<bool>("DataFormatClient",false);
  digiClient_ = ps.getUntrackedParameter<bool>("DigiClient",false);
  recHitClient_ = ps.getUntrackedParameter<bool>("RecHitClient",false);
  pedestalClient_ = ps.getUntrackedParameter<bool>("PedestalClient",false);
  ledClient_ = ps.getUntrackedParameter<bool>("LEDClient",false);
  hotCellClient_ = ps.getUntrackedParameter<bool>("HotCellClient",false);
  deadCellClient_ = ps.getUntrackedParameter<bool>("DeadCellClient",false);
  trigPrimClient_ = ps.getUntrackedParameter<bool>("TrigPrimClient",false);
  caloTowerClient_ = ps.getUntrackedParameter<bool>("CaloTowerClient",false);
  

  phiBins_=(int)(abs(phiMax_-phiMin_));
  etaBins_=(int)(abs(etaMax_-etaMin_));

  // Summary maps
  meGlobalSummary_=0;

  // All initial status floats set to -1 (uncertain)
  // For the moment, these are just local variables; if we want to keep
  // them in the root file, we need to book them as MonitorElements
  status_HB_=-1;
  status_HE_=-1;
  status_HO_=-1;
  status_HF_=-1;
  status_global_=-1;
  
  subdetCells_.insert(make_pair("HB",2592));
  subdetCells_.insert(make_pair("HE",2592));
  subdetCells_.insert(make_pair("HO",2160));
  subdetCells_.insert(make_pair("HF",1728));
  
} // HcalSummaryClient::HcalSummaryClient(const ParameterSet& ps)


HcalSummaryClient::~HcalSummaryClient()
{
}

void HcalSummaryClient::beginJob(DQMStore* dqmStore)
{
  dqmStore_=dqmStore;
  if (debug_) 
    cout <<"HcalSummaryClient: beginJob"<<endl;
  ievt_ = 0; // keepts track of all events in job
  jevt_ = 0; // keeps track of all events in run
  lastupdate_=0; // keeps analyze from being called by both endRun and endJob
} // void HcalSummaryClient::beginJob(DQMStore* dqmStore)


void HcalSummaryClient::beginRun(void)
{
  if ( debug_ ) cout << "HcalSummaryClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
} //void HcalSummaryClient::beginRun(void)


void HcalSummaryClient::endJob(void)
{
  if ( debug_ ) cout << "HcalSummaryClient: endJob, ievt = " << ievt_ << endl;
  // When the job ends, we want to make a summary before exiting
  if (ievt_>lastupdate_)
    analyze();
  this->cleanup();
} // void HcalSummaryClient::endJob(void)


void HcalSummaryClient::endRun(void) 
{
  if ( debug_ ) cout << "HcalSummaryClient: endRun, jevt = " << jevt_ << endl;
  // When the run ends, we want to make a summary before exiting
  analyze();
  lastupdate_=ievt_;
  this->cleanup();
} // void HcalSummaryClient::endRun(void) 


void HcalSummaryClient::setup(void)
{

  char histo[200];

  // Is this the correct folder?
  dqmStore_->setCurrentFolder( prefixME_ + "/HcalSummaryClient" );


  // This histogram may be redundant?
  sprintf(histo,"Global Summary");
  meGlobalSummary_ = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
					 phiBins_,phiMin_,phiMax_);
  meGlobalSummary_->setAxisTitle("i#eta", 1);
  meGlobalSummary_->setAxisTitle("i#phi", 2);


  
  // Monitor Elements in required format, according to https://twiki.cern.ch/twiki/bin/view/CMS/SummaryDisplayProposal
  MonitorElement* me;

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummary");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) {
    dqmStore_->removeElement(me->getName());
  }
  me = dqmStore_->bookFloat(histo);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );


  // Create floats showing subtasks status
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

  sprintf(histo,"Hcal_HB");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"Hcal_HE");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"Hcal_HO");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"Hcal_HF");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);


  // Create global summary map
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummaryMap");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) 
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
			 phiBins_,phiMin_,phiMax_);
  
  me->setAxisTitle("i#eta", 1);
  me->setAxisTitle("i#phi", 2);
 

  // set status words to unknown by default
  status_HB_=-1;
  status_HE_=-1;
  status_HO_=-1;
  status_HF_=-1;

  
  for (int i=0;i<4;++i)
    {
      // All values set to unknown by default
      status_digi[i]=-1;
      status_deadcell[i]=-1;
      status_hotcell[i]=-1;
    }


  // Get total number of cells from all subdetectors
  std::map<std::string, int>::const_iterator it;
  totalcells_=0;
  if (subDetsOn_[0])
    {
      it = subdetCells_.find("HB");
      totalcells_+=it->second;
    }
  if (subDetsOn_[1])
    {
      it = subdetCells_.find("HE");
      totalcells_+=it->second;
    }
  if (subDetsOn_[2])
    {
      it = subdetCells_.find("HO");
      totalcells_+=it->second;
    }
  if (subDetsOn_[3])
    {
      it = subdetCells_.find("HF");
      totalcells_+=it->second;
    }


} // void HcalSummaryClient::setup(void)


void HcalSummaryClient::cleanup(void) 
{
  
  if ( ! enableCleanup_ ) return;

  MonitorElement* me;

  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary") ) {
    dqmStore_->removeElement(me->getName());
  }
  

  /*
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap") ) {
    dqmStore_->removeElement(me->getName());
  }
  */  

  // redundant?  Handled above?
  if (meGlobalSummary_) dqmStore_->removeElement(meGlobalSummary_->getName());
  meGlobalSummary_=0;
  

} // void HcalSummaryClient::cleanup(void) 



void HcalSummaryClient::incrementCounters(void)
{
  ievt_++;
  jevt_++;
  return;
}

void HcalSummaryClient::analyze(void)
{

  if (debug_)
    cout <<"HcalSummaryClient:  Running analyze..."<<endl;
  if ( ievt_ % 10 == 0 ) 
    {
      if ( debug_ )
	cout << "HcalSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
    }

  // Reset values to 'unknown' status; they'll be set by analyze_everything routines
  status_global_=-1;
  status_HB_=-1;
  status_HE_=-1;
  status_HO_=-1;
  status_HF_=-1;

  // Reset summary map to 'unknown' status 
  MonitorElement* reportMap = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (!reportMap)
    {
      cout <<"<HcalSummaryClient::analyze> Could not get reportSummaryMap!"<<endl;
      return;
    }

  // Set all bins to "empty" to start (change to "unknown" (-1)?)
  for (int ieta=1;ieta<=etaBins_;++ieta)
    for (int iphi=1; iphi<=phiBins_;++iphi)
      reportMap->setBinContent(ieta,iphi,-1);


  // Now look for problem cells in each Task & Subdetector
  if (subDetsOn_[0]) analyze_everything("HB",1,status_HB_);
  if (subDetsOn_[1]) analyze_everything("HE",2,status_HE_);
  if (subDetsOn_[2]) analyze_everything("HO",3,status_HO_);
  if (subDetsOn_[3]) analyze_everything("HF",4,status_HF_);


  // Set the status words in the root file
  MonitorElement* me;
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me) me->Fill(status_global_);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HB") )
    me->Fill(status_HB_);						 
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HE") )
    me->Fill(status_HE_);						 
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HO") )
    me->Fill(status_HO_);						 
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/Hcal_HF") )
    me->Fill(status_HF_);
  
  dqmStore_->setCurrentFolder( prefixME_);


  return;
} //void HcalSummaryClient::analyze(void)


float HcalSummaryClient::analyze_everything(std::string subdetname, int type, float& subdet)
{
  /* analyze_everything calculates overall status from all available client checks */

  if (debug_) cout<<"<HcalSummaryClient> Running analyze_everything"<<endl;

  float status = -1; // default status value
  // Get 2D map (histogram) of known problems; return -1 if map not found
  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");

  if (!me)
    return status;
  
  char name[150];

  MonitorElement* me_digi=0;
  MonitorElement* me_hotcell=0;
  MonitorElement* me_deadcell=0;

  // Check for histogram containing known Digi problems.  (Formed & filled in DigiMonitor task)
  if (digiClient_)
    {
      sprintf(name,"%s/DigiMonitor/%s/%sProblemDigiCells",prefixME_.c_str(),
	      subdetname.c_str(),subdetname.c_str());  // form histogram name
      me_digi = dqmStore_->get(name);
      if (!me_digi) return status; // histogram couldn't be found
    }

  //Check for histogram containing known HotCell problems
  if (hotCellClient_)
    {
      sprintf(name,"%s/HotCellMonitor/%s/%sProblemHotCells",prefixME_.c_str(),
	      subdetname.c_str(),subdetname.c_str());
      me_hotcell = dqmStore_->get(name); // get Monitor Element named 'name'
      if (!me_hotcell) 
	{
	  if (debug_) cout <<"<HcalSummaryClient>  Could not get ProblemHotCells"<<endl;
	  return status;
	}
    }

  // Check for histogram containing known DeadCell problems
  if (deadCellClient_)
    {
      sprintf(name,"%s/DeadCellMonitor/%s/%sProblemDeadCells",prefixME_.c_str(),
	      subdetname.c_str(),subdetname.c_str());
      me_deadcell = dqmStore_->get(name); // get Monitor Element named 'name'
      if (!me_deadcell) return status;
    }

  // loop over cells
  double newbincontent=0;
  float badcells=0.; 
  float baddigis=0.;
  float badhotcells=0.;
  float baddeadcells=0.;
  int eta;
  int phi;

  float tempval=0.;
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {   
      eta=ieta+int(etaMin_)-1;
      if (eta==0) continue; // skip eta=0 bin -- unphysical
      if (abs(eta)>41) continue; // skip unphysical "boundary" bins in histogram
      // IMPORTANT:  HB = type1, HE=2, HO=3, HF=4
      if (type==1 && abs(eta)>16) continue;
      if (type==2 && (abs(eta)<16 || abs(eta)>29)) continue;
      if (type==4 && abs(eta)<29) continue;
      if (type==3 && abs(eta)>15) continue;


      for (int iphi=1; iphi<=phiBins_;++iphi)
	{
	  phi=iphi+int(phiMin_)-1;
	  // Skip non-physical cells
	  if (phi<1 || phi>72) continue;
	  if (abs(eta)>20 && phi%2==0) continue;
	  if (abs(eta)>39 && phi%4!=3) continue;

	  newbincontent=0;
	  // Now check for errors from each client histogram:
	  if (digiClient_) 
	    {
	      tempval=me_digi->getBinContent(ieta,iphi);
	      baddigis+=tempval;
	      newbincontent+=tempval;

	    }
	  if (hotCellClient_) 
	    {
	      tempval=me_hotcell->getBinContent(ieta,iphi);
	      badhotcells+=tempval;
	      newbincontent+=tempval;
	    }
	  if (deadCellClient_) 
	    {
	      tempval=me_deadcell->getBinContent(ieta,iphi);
	      baddeadcells+=tempval;
	      newbincontent+=tempval;
	    }
	   
	  //if (newbincontent==0) continue; 
	   
	  
	  newbincontent/=ievt_; // normalize to number of events
	  if (newbincontent>0)
	    {
	      //cout <<"BAD BIN "<<ieta<<"  "<< iphi<<"  ("<<eta<<", "<<phi<<") "<<newbincontent<<endl;
	      badcells+=newbincontent;
	    }
	  
	  if (newbincontent>1) newbincontent=1;  // bad bin content represents fraction of bad events; can't have value >1 (even where there are multiple depths for a given eta,phi coordinate)
	  
	  // ReportSummaryMap should show good cells as "1", bad as "0" -- use 1-newbincontent
	  // See if value has already been filled from other subdetector
	  tempval=me->getBinContent(ieta,iphi);
	  if (tempval==-1)
	    me->setBinContent(ieta,iphi,1-newbincontent);
	  else
	    {
	      tempval=tempval-newbincontent;
	      if (tempval<0) tempval=0;
	      //if (tempval<=0) tempval=0.00001;
	      me->setBinContent(ieta,iphi,tempval);
	    }
	} // loop over iphi
    } // loop over ieta


// subdetCells_ stores number of cells in each subdetector
  std::map<std::string, int>::const_iterator it;
  
  it =subdetCells_.find(subdetname);

  // didn't find subdet in map -- return -1
  if (it==subdetCells_.end())
    return -1;

  // Map claims that subdetector has no cells, or fewer cells than # of bad cells found:  return -1
  if (it->second == 0 || (it->second)<badcells)
    return -1;

  // Normalize by number of events
  baddigis/=ievt_;
  badhotcells/=ievt_;
  baddeadcells/=ievt_;
  
  // Status is 1 if no bad cells found
  // Otherwise, status = 1 - (avg fraction of bad cells/event)
  status_digi[type-1]=1.-(1.*baddigis)/it->second;
  status_deadcell[type-1]=1.-(1.*baddeadcells)/it->second;
  status_hotcell[type-1]=1.-(1.*badhotcells)/it->second;
 
  // overall status is product of individual statuses.  Is this what we want?
  status=status_digi[type-1]*status_deadcell[type-1]*status_hotcell[type-1];
  // status = 1. - (1.*badcells)/it->second;  // old definition

  // Set subdetector status to 'status' value
  subdet=status;

  // Old status version: Global status set by multiplying subdetector statuses
  /*
  if (status_global_==-1)
    status_global_=status;
  else
    status_global_*=status;
  */

  // New version:  status is average over all cells (this means scaling each subdetector status value by its fractional contribution (it->second)/totalcells_ )
  if (status_global_==-1)
    status_global_=1.*status*(it->second)/totalcells_;
  else
    status_global_+=1.*status*(it->second)/totalcells_;

  //cout <<subdetname.c_str()<<" SUBDET STATUS = "<<status<<"  GLOBAL = "<<status_global_<<"   (scale factor = "<<(it->second)<<"/"<<totalcells_<<" = "<<(1.*(it->second)/totalcells_)<<")"<<endl;

  return status;
} // float HcalSummaryClient::analyze_everything





void HcalSummaryClient::htmlOutput(int& run, time_t& mytime, int& minlumi, int& maxlumi, string& htmlDir, string& htmlName)
{

  //if ( verbose_ ) 
  cout << "Preparing HcalSummaryClient html output ..." << endl;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span>" << endl;

  std::string startTime=ctime(&mytime);
  htmlFile << "&nbsp;&nbsp;LS:&nbsp;" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << minlumi << "</span>" << endl;
  htmlFile << "-" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << maxlumi << "</span>" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Start&nbsp;Time:&nbsp;<spa\n style=\"color: rgb(0, 0, 153);\">" << startTime << "</span></h2> " << endl;
  
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span> </h2> " << endl;


  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;

  /*
    htmlFile << "<hr>" << endl;
    htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
    htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
    htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;
  */

  // Produce the plots to be shown as .png files from existing histograms

  // values taken from EBSummaryClient.cc
  //const int csize = 400;

  //TCanvas* cMap = new TCanvas("cMap", "Temp", int(360./170.*csize), csize);
  //TCanvas* cMapPN = new TCanvas("cMapPN", "Temp", int(360./170.*csize), int(20./90.*360./170.*csize));

  //  const double histMax = 1.e15;

  TH2F* obj2f;
  std::string imgNameMap="";
  std::string imgName;
  gStyle->SetPaintTextFormat("+g");

  std::string meName;


  // Test for now -- let's just dump out global summary histogram
  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  obj2f = me->getTH2F();

  // borrowing palette code from 

  static int pcol[20];
  float rgb[20][3];
  
  for( int i=0; i<20; ++i ) 
    {
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
      pcol[i] = 901+i;
      TColor* color = gROOT->GetColor( 901+i );
      if( ! color ) color = new TColor( 901+i, 0, 0, 0, "" );
      color->SetRGB( rgb[i][0], rgb[i][1], rgb[i][2] );
    } // for (int i=0;i<20;++i)
 
   gStyle->SetPalette(20, pcol);
 
   if( obj2f ) 
     {
       obj2f->SetMinimum(-1.e-15);
       obj2f->SetMaximum(+1.0);
       obj2f->SetOption("colz");
     }



  if (obj2f)// && obj2f->GetEntries()!=0)
    {
      htmlFile << "<table  width=100% border=1><tr>" << endl; 
      htmlFile << "<tr align=\"center\">" << endl;  
      htmlAnyHisto(run,obj2f,"i#eta","i#phi",92,htmlFile,htmlDir);
      htmlFile <<"</tr></table><hr>"<<endl;

    } // if (obj2f)

  
  // Make table that lists all status words for each subdet
  
  htmlFile<<"<hr><br><h2>Summary Values for Each Subdetector</h2><br>"<<endl;
  htmlFile << "<table border=\"2\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile <<"<td>Task</td><td>HB</td><td>HE</td><td>HO</td><td>HF</td><td>HCAL</td></tr>"<<endl;

  vector<string> subdets;
  subdets.push_back("HB");
  subdets.push_back("HE");
  subdets.push_back("HO");
  subdets.push_back("HF");

  if (digiClient_)
    {
      float tempstatus=-1;
      for (int i=0;i<4;++i)
	{
	  if (status_digi[i]>-1)
	    {
	      std::map<std::string, int>::const_iterator it;
	      it = subdetCells_.find(subdets[i]);
	      (tempstatus==-1) ?  tempstatus=status_digi[i]*(it->second)/totalcells_ : tempstatus+=status_digi[i]*(it->second)/totalcells_;
	    }
	}

      htmlFile <<"<tr><td>Digi Monitor</td>"<<endl;
      subDetsOn_[0]==1 ? htmlFile<<"<td>"<<status_digi[0]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[1]==1 ? htmlFile<<"<td>"<<status_digi[1]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[2]==1 ? htmlFile<<"<td>"<<status_digi[2]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[3]==1 ? htmlFile<<"<td>"<<status_digi[3]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      htmlFile<<"<td>"<< tempstatus  <<"</td>"<<endl;
      htmlFile<<"</tr>"<<endl;
    }

  if (deadCellClient_)
    {
      float tempstatus=-1;
      for (int i=0;i<4;++i)
	{
	  if (status_deadcell[i]>-1)
	    { 
	      std::map<std::string, int>::const_iterator it;
	      it = subdetCells_.find(subdets[i]);
	      (tempstatus==-1) ?  tempstatus=status_deadcell[i]*(it->second)/totalcells_ : tempstatus+=status_deadcell[i]*(it->second)/totalcells_;

	    }
	}
      
      htmlFile <<"<tr><td>Dead Cell Monitor</td>"<<endl;
      subDetsOn_[0]==1 ? htmlFile<<"<td>"<<status_deadcell[0]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[1]==1 ? htmlFile<<"<td>"<<status_deadcell[1]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[2]==1 ? htmlFile<<"<td>"<<status_deadcell[2]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[3]==1 ? htmlFile<<"<td>"<<status_deadcell[3]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      htmlFile<<"<td>"<< tempstatus  <<"</td>"<<endl;
      htmlFile<<"</tr>"<<endl;
    }
  
  
  if (hotCellClient_)
    {
	
      float tempstatus=-1;
      for (int i=0;i<4;++i)
	{
	  if (status_hotcell[i]>-1)
	    {
	      std::map<std::string, int>::const_iterator it;
	      it = subdetCells_.find(subdets[i]);
	      (tempstatus==-1) ?  tempstatus=status_hotcell[i]*(it->second)/totalcells_ : tempstatus+=status_hotcell[i]*(it->second)/totalcells_;


	    }
	}

      htmlFile <<"<tr><td>Hot Cell Monitor</td>"<<endl;
      subDetsOn_[0]==1 ? htmlFile<<"<td>"<<status_hotcell[0]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[1]==1 ? htmlFile<<"<td>"<<status_hotcell[1]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[2]==1 ? htmlFile<<"<td>"<<status_hotcell[2]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      subDetsOn_[3]==1 ? htmlFile<<"<td>"<<status_hotcell[3]<<"</td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
      htmlFile<<"<td>"<< tempstatus  <<"</td>"<<endl;
      htmlFile<<"</tr>"<<endl;
    }

  // Dump out final status words

  htmlFile <<"<tr><td><font color = \"blue\"> Status Values </font></td>"<<endl;
  subDetsOn_[0]==1 ? htmlFile<<"<td><font color = \"blue\">"<<status_HB_<<"</font></td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
  subDetsOn_[1]==1 ? htmlFile<<"<td><font color = \"blue\">"<<status_HE_<<"</font></td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
  subDetsOn_[2]==1 ? htmlFile<<"<td><font color = \"blue\">"<<status_HO_<<"</font></td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
  subDetsOn_[3]==1 ? htmlFile<<"<td><font color = \"blue\">"<<status_HF_<<"</font></td>"<<endl : htmlFile<<"<td>--</td>"<<endl;
  htmlFile <<"<td><font color = \"blue\">"<<status_global_<<"</font></td>"<<endl;
  htmlFile <<"</tr></table>"<<endl;

  htmlFile <<"<br><h2> A note on Status Values </h2><br>"<<endl;
  htmlFile <<"Status values in each subdetector task represent the average fraction of good channels per event.  (For example, a value of .99 in the HB Hot Cell monitor means that, on average, 1% of the cells in HB are hot.)  Status values should range from 0 to 1, with a perfectly-functioning detector will have all status values = 1.  If the status is unknown, a value of -1 or \"--\" will be shown. <br>"<<endl;
  htmlFile <<"The HCAL status values for each task are a weighted average from each subdetector.  Weights are assigned as (# of cells in the subdetector)/(total # of cells being checked).<br>"<<endl;
  htmlFile <<"The overall Status Values at the bottom of the table are the product of each of the individual Monitor values.  These values are not quite the same as the overall fraction of good channels in an event.  (For instance, if the Digi Monitor and Dead Cell Monitor both had a status of 0.5, the overall status would then be 0.25.  This does not mean that only 25% of the cells in HCAL are bad, because the Digi and Dead Cell Monitors could be complaining about different cells.  The overall Status Values thus represent a 'worst case' scenario, assuming that all Task problems are independent.<br>"<<endl;


  /*
<br>The HCAL values are the product of the individual subdetectors, and so these status values don't represent the average number of bad cells in the entire detector.  (For instance, if both HB and HF Hot Cell monitors had status values of 0.5, the overall HCAL status value would be 0.25.  This is not the same as saying that 3/4 of the cells in HCAL are hot.)<br><br>  Likewise, the overall status values are formed by the products of the individual tasks.  The overall status values thus provide some measure of the number of bad cells per event, but they do not represent the average number of good channels per event.<br>"<<endl;
  */
  htmlFile <<"<br><hr><br>"<<endl;
  htmlFile <<"Run #: "<<run<<"&nbsp;&nbsp;&nbsp;&nbsp Starting Time: "<<startTime<<"&nbsp;&nbsp;&nbsp;&nbsp;";
  htmlFile<<"Luminosity blocks: "<<minlumi<<" - "<<maxlumi <<"&nbsp;&nbsp;&nbsp;&nbsp";
  htmlFile <<"# of events: "<<ievt_<<"&nbsp;&nbsp;&nbsp;&nbsp";
  
  htmlFile <<"  Digi Status:  HB: "<<status_digi[0]<<" HE: "<<status_digi[1]<<" HO: "<<status_digi[2]<<" HF: "<<status_digi[3]<<"&nbsp;&nbsp;&nbsp;&nbsp";
  htmlFile <<"  DeadCell Status:  HB: "<<status_deadcell[0]<<" HE: "<<status_deadcell[1]<<" HO: "<<status_deadcell[2]<<" HF: "<<status_deadcell[3]<<"&nbsp;&nbsp;&nbsp;&nbsp";
  htmlFile <<"  HotCell Status:  HB: "<<status_hotcell[0]<<" HE: "<<status_hotcell[1]<<" HO: "<<status_hotcell[2]<<" HF: "<<status_hotcell[3]<<"&nbsp;&nbsp;&nbsp;&nbsp";
  htmlFile <<"  OVERALL STATUS:  "<<status_global_<<endl;
  htmlFile.close();

} // void HcalSummaryClient::htmlOutput(int run, string& htmlDir, string& htmlName)



