#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

/*
 * \file HcalSummaryClient.cc
 * 
 * $Date: 2010/03/03 20:02:52 $
 * $Revision: 1.64.2.4 $
 * \author J. Temple
 * \brief Summary Client class
 */

using namespace std;
using namespace edm;

HcalSummaryClient::HcalSummaryClient(std::string myname)
{
  name_=myname;
  SummaryMapByDepth=0;
  minevents_=0;
  minerrorrate_=0;
  badChannelStatusMask_=0;
}

HcalSummaryClient::HcalSummaryClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<string>("SummaryFolder","EventInfo/"); // SummaryMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  NLumiBlocks_ = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  // These aren't used in summary client, are they?
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("Summary_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  minerrorrate_ = ps.getUntrackedParameter<double>("Summary_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0));
  minevents_    = ps.getUntrackedParameter<int>("Summary_minevents",
						ps.getUntrackedParameter<int>("minevents",0));
  SummaryMapByDepth=0;
}

void HcalSummaryClient::analyze(int LS)
{
  if (debug_>2) std::cout <<"\tHcalSummaryClient::analyze()"<<std::endl;

  // 

  // Start with counters in 'unknown' status; they'll be set by analyze_everything routines 
  status_global_=-1; 
  status_HB_=-1; 
  status_HE_=-1; 
  status_HO_=-1; 
  status_HF_=-1; 

  status_HO0_=-1;
  status_HO12_=-1;
  status_HFlumi_=-1;
  status_global_=-1;

  EnoughEvents_->Reset();
  enoughevents_=true; // assume we have enough events for all tests to have run
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    {
      cout <<"CLIENT = "<<clients_[i]->name_<<"  ENOUGH = "<<clients_[i]->enoughevents_<<endl;
      enoughevents_&=clients_[i]->enoughevents_;
      EnoughEvents_->setBinContent(i+1,clients_[i]->enoughevents_);
      {
	if (clients_[i]->enoughevents_==false && debug_>1)
	  std::cout <<"Failed enoughevents test for monitor "<<clients_[i]->name()<<std::endl;
      }
    }
  if (enoughevents_==false)
    {
      if (debug_>0) std::cout <<"<HcalSummaryClient::analyze>  Not enough events processed to evaluate summary status!"<<std::endl;
      fillReportSummary(LS);
      return;
    }
  EnoughEvents_->setBinContent(clients_.size()+1,1); // summary is good to go!

  // check to find which subdetectors are present
  MonitorElement* temp_present;
  if (HBpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HBpresent");
      if (temp_present!=0)
	HBpresent_=temp_present->getIntValue();
    }
  if (HEpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HEpresent");
      if (temp_present!=0)
	HEpresent_=temp_present->getIntValue();
    }
  if (HOpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HOpresent");
      if (temp_present!=0)
	HOpresent_=temp_present->getIntValue();
    }
  if (HFpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HFpresent");
      if (temp_present!=0)
	HFpresent_=temp_present->getIntValue();
    }

  if (debug_>1) 
   std::cout <<"<HcalSummaryClient::analyze>  HB present = "<<HBpresent_<<" "<<"HE present = "<<HEpresent_<<" "<<"HO present = "<<HOpresent_<<" "<<"HF present = "<<HFpresent_<<std::endl;

 // set status to 0 if subdetector is present (or assumed present)
 if (HBpresent_!=0) status_HB_=0;
 if (HEpresent_!=0) status_HE_=0;
 if (HOpresent_!=0) {status_HO_=0; status_HO0_=0; status_HO12_=0;}
 if (HFpresent_!=0) {status_HF_=0; status_HFlumi_=0;}

 if (HBpresent_!=0 || HEpresent_!=0 ||
     HOpresent_!=0 || HFpresent_!=0 ) 
   status_global_=0;

 // reset all depth histograms
 if (SummaryMapByDepth==0 && debug_>0)
   std::cout <<"<HcalSummaryClient::analyze>  ERROR:  SummaryMapByDepth can't be found!"<<std::endl;
 else 
   {
     for (unsigned int i=0;i<(SummaryMapByDepth->depth).size();++i)
       SummaryMapByDepth->depth[i]->Reset();
     
     int etabins=-9999;
     int phibins=-9999;
     for (int d=0;d<4;++d)
       {
	 etabins=(SummaryMapByDepth->depth[d])->getNbinsX();
	 phibins=(SummaryMapByDepth->depth[d])->getNbinsY();
	 for (int eta=1;eta<=etabins;++eta)
	   {
	     int ieta=CalcIeta(eta-1,d+1);
	     for (int phi=1;phi<=phibins;++phi)
	       {
		 // loop over all client tests
		 for (unsigned int cl=0;cl<clients_.size();++cl)
		   {
		     // Best way to handle this?  
		     // We know that first element is HcalMonitorModule info, which has
		     // no problem cells defined.  Create some, or start counting from cl=1?
		     if (debug_>4 && eta==1 && phi==1) std::cout <<"Checking summary for client "<<clients_[cl]->name()<<std::endl;
		     if (clients_[cl]->ProblemCellsByDepth==0) continue;

		     if ((clients_[cl]->ProblemCellsByDepth)->depth[d]==0) continue;
		     if ((clients_[cl]->ProblemCellsByDepth)->depth[d]->getBinContent(eta,phi)>clients_[cl]->minerrorrate_)
		       {
			 if ((clients_[cl]->ProblemCellsByDepth)->depth[d]->getBinContent(eta,phi)<999)
			   SummaryMapByDepth->depth[d]->setBinContent(eta,phi,1);
			 else 
			   SummaryMapByDepth->depth[d]->setBinContent(eta,phi,2); // known problems filled with a value of 2
			 if (isHF(eta-1,d+1)) 
			   {
			     ++status_HF_;
			     if ((d==0 && (abs(ieta)==33 || abs(ieta)==34)) ||   // depth 1, rings 33,34
				 (d==1 && (abs(ieta)==35 || abs(ieta)==36)))     // depth 2, rings 35,36
			       ++status_HFlumi_; 
			   }
			 else if (isHO(eta-1,d+1)) 
			   {
			     ++status_HO_;
			     if (abs(ieta)<5) ++status_HO0_;
			     else ++status_HO12_;
			   }
			 else if (isHB(eta-1,d+1)) ++status_HB_;
			 else if (isHE(eta-1,d+1)) ++status_HE_;
			 break;
		       }
		   }
	       }
	   }
       } // for (int d=0;d<4;++d)
   } // else (SummaryMapByDepth exists)

 // We've checked all problems; now compute overall status
 int totalcells=0;
 std::map<std::string, int>::const_iterator it;

 if (HBpresent_!=0)
   {
     status_global_+=status_HB_; 
     it=subdetCells_.find("HB");
     totalcells+=it->second;
     status_HB_= 1-(status_HB_/it->second);
     status_HB_=max(0.,status_HB_); // converts fraction of bad channels to good fraction
   }
 else status_HB_=-1;

 if (HEpresent_!=0)
   {
     status_global_+=status_HE_;
     it=subdetCells_.find("HE");
     totalcells+=it->second;
     status_HE_= 1-(status_HE_/it->second);
     status_HE_=max(0.,status_HE_); // converts fraction of bad channels to good fraction
   }
 else status_HE_=-1;
 
 if (HOpresent_!=0)
   {
     status_global_+=status_HO_;
     it=subdetCells_.find("HO");
     totalcells+=it->second;
     status_HO_= 1-(status_HO_/it->second);
     status_HO_=max(0.,status_HO_); // converts fraction of bad channels to good fraction
     
     it=subdetCells_.find("HO0");
     status_HO0_= 1-(status_HO0_/it->second);
     status_HO0_=max(0.,status_HO0_); // converts fraction of bad channels to good fraction
     it=subdetCells_.find("HO12");
     status_HO12_= 1-(status_HO12_/it->second);
     status_HO12_=max(0.,status_HO12_); // converts fraction of bad channels to good fraction
   }
 else
   {
     status_HO_=-1;
     status_HO0_=-1;
     status_HO12_=-1;
   }
  if (HFpresent_!=0)
    {
      status_global_+=status_HF_;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      status_HF_= 1-(status_HF_/it->second);
      status_HF_=max(0.,status_HF_); // converts fraction of bad channels to good fraction
      it=subdetCells_.find("HFlumi");
      status_HFlumi_= 1-(status_HFlumi_/it->second);
      status_HFlumi_=max(0.,status_HFlumi_); // converts fraction of bad channels to good fraction
    }
  else
    {
      status_HF_=-1;
      status_HFlumi_=-1;
    }

  if (totalcells==0)
    status_global_=-1;
  else
    {
      status_global_=1-status_global_/totalcells;
      status_global_=max(0.,status_global_); // convert to good fraction
    }
  fillReportSummary(LS);
} // analyze

void HcalSummaryClient::fillReportSummary(int LS)
{

 // We've now checked all tasks; now let's calculate summary values
 
  if (debug_>2)  std::cout <<"<HcalSummaryClient::fillReportSummary>"<<std::endl;

  if (debug_>3) 
    {
      std::cout <<"STATUS = "<<endl;
      std:: cout <<"HB = "<<status_HB_<<endl;
      std:: cout <<"HE = "<<status_HE_<<endl;
      std:: cout <<"HO = "<<status_HO_<<endl;
      std:: cout <<"HF = "<<status_HF_<<endl;
      std:: cout <<"HO0 = "<<status_HO0_<<endl;
      std:: cout <<"HO12 = "<<status_HO12_<<endl;
      std:: cout <<"HFlumi = "<<status_HFlumi_<<endl;
    }

  // put the summary values into MonitorElements 

  if (LS>0)
    {
      StatusVsLS_->setBinContent(LS,1,status_HB_);
      StatusVsLS_->setBinContent(LS,2,status_HE_);
      StatusVsLS_->setBinContent(LS,3,status_HO_);
      StatusVsLS_->setBinContent(LS,4,status_HF_);
      StatusVsLS_->setBinContent(LS,5,status_HO0_);
      StatusVsLS_->setBinContent(LS,6,status_HO12_);
      StatusVsLS_->setBinContent(LS,7,status_HFlumi_);
    }


  MonitorElement* me;
  dqmStore_->setCurrentFolder(subdir_);
 
  //me=dqmStore_->get(subdir_+"reportSummaryMap");
  if (reportMap_)
    {
      reportMap_->setBinContent(1,1,status_HB_);
      reportMap_->setBinContent(2,1,status_HE_);
      reportMap_->setBinContent(3,1,status_HO_);
      reportMap_->setBinContent(4,1,status_HF_);
      reportMap_->setBinContent(5,1,status_HO0_);
      reportMap_->setBinContent(6,1,status_HO12_);
      reportMap_->setBinContent(7,1,status_HFlumi_);
    }
  else if (debug_>0) std::cout <<"<HcalSummaryClient::fillReportSummary> CANNOT GET REPORT SUMMARY MAP!!!!!"<<std::endl;

  me=dqmStore_->get(subdir_+"reportSummary");
  // Clear away old versions
  if (me) me->Fill(status_global_);

  // Create floats for each subdetector status
  std::string subdets[7] = {"HB","HE","HO","HF","HO0","HO12","HFlumi"};
  for (unsigned int i=0;i<7;++i)
    {
      // Create floats showing subtasks status
      dqmStore_->setCurrentFolder( subdir_+ "reportSummaryContents" );  
      me=dqmStore_->get(subdir_+"reportSummaryContents/Hcal_"+subdets[i]);
      if (me==0)
	{
	  if (debug_>0) cout <<"<HcalSummaryClient::analyze()>  Could not get Monitor Element named 'Hcal_"<<subdets[i]<<"'"<<std::endl;
	  continue;
	}
      if (subdets[i]=="HB") me->Fill(status_HB_);
      else if (subdets[i]=="HE") me->Fill(status_HE_);
      else if (subdets[i]=="HO") me->Fill(status_HO_);
      else if (subdets[i]=="HF") me->Fill(status_HF_);
      else if (subdets[i]=="HO0") me->Fill(status_HO0_);
      else if (subdets[i]=="HO12") me->Fill(status_HO12_);
      else if (subdets[i]=="HFlumi") me->Fill(status_HFlumi_);
    } // for (unsigned int i=0;...)

} // fillReportSummary()


void HcalSummaryClient::beginJob()
{
  dqmStore_ = Service<DQMStore>().operator->();
  // set total number of cells in each subdetector
  subdetCells_.insert(make_pair("HB",2592));
  subdetCells_.insert(make_pair("HE",2592));
  subdetCells_.insert(make_pair("HO",2160));
  subdetCells_.insert(make_pair("HF",1728));
  subdetCells_.insert(make_pair("HO0",576));
  subdetCells_.insert(make_pair("HO12",1584));
  subdetCells_.insert(make_pair("HFlumi",288));  // 8 rings, 36 cells/ring
  // Assume subdetectors are 'unknown'
  HBpresent_=-1;
  HEpresent_=-1;
  HOpresent_=-1;
  HFpresent_=-1;
  
  EnoughEvents_=0;
  MinEvents_=0;
  MinErrorRate_=0;

}

void HcalSummaryClient::endJob(){}

void HcalSummaryClient::beginRun(void)
{
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalSummaryClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  nevts_=0;

  dqmStore_->setCurrentFolder(subdir_);

  MonitorElement* me;
  // reportSummary holds overall detector status
  me=dqmStore_->get(subdir_+"reportSummary");
  // Clear away old versions
  if (me) dqmStore_->removeElement(me->getName());
  me = dqmStore_->bookFloat("reportSummary");
  me->Fill(-1); // set status to unknown at startup

  // Create floats for each subdetector status
  std::string subdets[7] = {"HB","HE","HO","HF","HO0","HO12","HFlumi"};
  for (unsigned int i=0;i<7;++i)
    {
      // Create floats showing subtasks status
      dqmStore_->setCurrentFolder( subdir_+ "reportSummaryContents" );  
      me=dqmStore_->get(subdir_+"reportSummaryContents/Hcal_"+subdets[i]);
      if (me) dqmStore_->removeElement(me->getName());
      me = dqmStore_->bookFloat("Hcal_"+subdets[i]);
      me->Fill(-1);
    } // for (unsigned int i=0;...)

  dqmStore_->setCurrentFolder(prefixME_+"HcalInfo/SummaryClientPlots");
  me=dqmStore_->get(prefixME_+"HcalInfo/SummaryClientPlots/HB HE HF Depth 1 Problem Summary Map");
  if (me) dqmStore_->removeElement(me->getName());
  me=dqmStore_->get(prefixME_+"HcalInfo/SummaryClientPlots/HB HE HF Depth 2 Problem Summary Map");
  if (me) dqmStore_->removeElement(me->getName());
  me=dqmStore_->get(prefixME_+"HcalInfo/SummaryClientPlots/HE Depth 3 Problem Summary Map");
  if (me) dqmStore_->removeElement(me->getName());
   me=dqmStore_->get(prefixME_+"HcalInfo/SummaryClientPlots/HO Depth 4 Problem Summary Map");
  if (me) dqmStore_->removeElement(me->getName());

  if (EnoughEvents_==0)
    EnoughEvents_=dqmStore_->book1D("EnoughEvents","Enough Events Passed From Each Task To Form Summary?",1+(int)clients_.size(),0,1+(int)clients_.size());
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    EnoughEvents_->setBinLabel(i+1,clients_[i]->name());
  EnoughEvents_->setBinLabel(1+(int)clients_.size(),"Summary");

  if (MinEvents_==0)
    MinEvents_=dqmStore_->book1D("MinEvents","Minimum Events Required From Each Task To From Summary",
				 1+(int)clients_.size(),0,1+(int)clients_.size());
  int summin=0;
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    {
      MinEvents_->setBinLabel(i+1,clients_[i]->name());
      MinEvents_->setBinContent(i+1,clients_[i]->minevents_);
      summin=max(summin,clients_[i]->minevents_);
    }
  if (MinErrorRate_==0)
    MinErrorRate_=dqmStore_->book1D("MinErrorRate",
				    "Minimum Error Rate Required For Channel To Be Counted As Problem",
				    (int)clients_.size(),0,(int)clients_.size());
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    {
      MinErrorRate_->setBinLabel(i+1,clients_[i]->name());
      MinErrorRate_->setBinContent(i+1,clients_[i]->minerrorrate_);
    }

  if (SummaryMapByDepth==0) 
    {
      SummaryMapByDepth=new EtaPhiHists();
      SummaryMapByDepth->setup(dqmStore_,"Problem Summary Map");
    }
  // Set histogram values to -1
  // Set all bins to "unknown" to start
  int etabins=0;
  for (unsigned int depth=0;depth<4;++depth)
    {
      if (SummaryMapByDepth->depth[depth]==0) continue;
      SummaryMapByDepth->depth[depth]->Reset();
      etabins=(SummaryMapByDepth->depth[depth])->getNbinsX();
      for (int ieta=0;ieta<etabins;++ieta)
	{
	  for (int iphi=0;iphi<72;++iphi)
	    SummaryMapByDepth->depth[depth]->setBinContent(ieta+1,iphi+1,-1);
	}
    }

  // Make histogram of status vs LS
  StatusVsLS_ = dqmStore_->get(prefixME_+"HcalInfo/SummaryClientPlots/StatusVsLS");
  if (StatusVsLS_) dqmStore_->removeElement(StatusVsLS_->getName());
  StatusVsLS_ = dqmStore_->book2D("StatusVsLS","Status vs. Luminosity Section",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 7,0,7);
  // Set all status values to -1 to begin
  for (int i=1;i<=NLumiBlocks_;++i)
    for (int j=1;j<=7;++j)
      StatusVsLS_->setBinContent(i,j,-1);
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(1,"HB");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(2,"HE");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(3,"HO");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(4,"HF");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(5,"HO0");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(6,"HO12");
  (StatusVsLS_->getTH2F())->GetYaxis()->SetBinLabel(7,"HFlumi");
  (StatusVsLS_->getTH2F())->GetXaxis()->SetTitle("Lumi Section");
  (StatusVsLS_->getTH2F())->SetMinimum(-1);
  (StatusVsLS_->getTH2F())->SetMaximum(1);

  // Finally, form report Summary Map
  dqmStore_->setCurrentFolder(subdir_);

  reportMap_=dqmStore_->get(subdir_+"reportSummaryMap");
  if (reportMap_)
    dqmStore_->removeElement(reportMap_->getName());
  reportMap_ = dqmStore_->book2D("reportSummaryMap","reportSummaryMap",
				 7,0,7,1,0,1);
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(1,"HB");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(2,"HE");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(3,"HO");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(4,"HF");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(5,"HO0");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(6,"HO12");
  (reportMap_->getTH2F())->GetXaxis()->SetBinLabel(7,"HFlumi");
  (reportMap_->getTH2F())->GetYaxis()->SetBinLabel(1,"Status");
  (reportMap_->getTH2F())->SetMarkerSize(3);
  (reportMap_->getTH2F())->SetOption("text90colz");
  (reportMap_->getTH2F())->SetOption("textcolz");
  (reportMap_->getTH2F())->SetMinimum(-1);
  (reportMap_->getTH2F())->SetMaximum(1);

  // Set initial counters to -1 (unknown)
  status_global_=-1; 
  status_HB_=-1; 
  status_HE_=-1; 
  status_HO_=-1; 
  status_HF_=-1; 

  status_HO0_=-1;
  status_HO12_=-1;
  status_HFlumi_=-1;
} // void HcalSummaryClient::beginRun(void)


void HcalSummaryClient::endRun(void){}

void HcalSummaryClient::setup(void){}
void HcalSummaryClient::cleanup(void){}

bool HcalSummaryClient::hasErrors_Temp(void){  return false;}

bool HcalSummaryClient::hasWarnings_Temp(void){return false;}
bool HcalSummaryClient::hasOther_Temp(void){return false;}
bool HcalSummaryClient::test_enabled(void){return true;}

void HcalSummaryClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual){return;}


