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
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.107 $
 * \author J. Temple
 * \brief Summary Client class
 */

HcalSummaryClient::HcalSummaryClient(std::string myname)
{
  name_=myname;
  SummaryMapByDepth=0;
  minevents_=0;
  minerrorrate_=0;
  badChannelStatusMask_=0;
  ProblemCells=0;
  ProblemCellsByDepth=0;
  StatusVsLS_=0;
  certificationMap_=0;
  reportMap_=0;
  reportMapShift_=0;
}

HcalSummaryClient::HcalSummaryClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("SummaryFolder","EventInfo/"); // SummaryMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  NLumiBlocks_ = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  UseBadChannelStatusInSummary_ = ps.getUntrackedParameter<bool>("UseBadChannelStatusInSummary",false);

  // These aren't used in summary client, are they?
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("Summary_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  minerrorrate_ = ps.getUntrackedParameter<double>("Summary_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0));
  minevents_    = ps.getUntrackedParameter<int>("Summary_minevents",
						ps.getUntrackedParameter<int>("minevents",0));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  SummaryMapByDepth=0;
  ProblemCells=0;
  ProblemCellsByDepth=0;
  StatusVsLS_=0;
  certificationMap_=0;
  reportMap_=0;
  reportMapShift_=0;
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

  if (EnoughEvents_!=0) EnoughEvents_->Reset();
  enoughevents_=true; // assume we have enough events for all tests to have run
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    {
      if (debug_>2) std::cout <<"<HcalSummaryClient::analyze>  CLIENT = "<<clients_[i]->name_<<"  ENOUGH = "<<clients_[i]->enoughevents_<<std::endl;
      enoughevents_&=clients_[i]->enoughevents_;
      if (EnoughEvents_!=0) EnoughEvents_->setBinContent(i+1,clients_[i]->enoughevents_);
      {
	if (clients_[i]->enoughevents_==false && debug_>1)
	  std::cout <<"Failed enoughevents test for monitor "<<clients_[i]->name()<<std::endl;
      }
    }

  // check to find which subdetectors are present -- need to do this prior to checking whether enoughevents_ == false!
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
  
  if (enoughevents_==false)
    {
      if (debug_>0) std::cout <<"<HcalSummaryClient::analyze>  Not enough events processed to evaluate summary status!"<<std::endl;
      
      // 'HXpresent_' values are set to -1 by default. 
      // They are set to +1 when a channel is present.
      // I don't think there are any cases where values =0,
      // but I'm not positive of this yet -- Jeff, 10 Aug 2010

      // Check whether any events are found for each subdetector
      if (HBpresent_>0) status_HB_=1;
      else status_HB_=-1;  // HB not present or unknown
      if (HEpresent_>0) status_HE_=1;
      else status_HE_=-1;  // HE not present or unknown
      if (HOpresent_>0) status_HO_=1;
      else status_HO_=-1;  // HO not present or unknown
      if (HFpresent_>0) status_HF_=1;
      else status_HF_=-1;  // HF not present or unknown

      // Update this in the future?  Use '||' instead of '&&'?
      if (HBpresent_<=0 && HEpresent_<=0 && HOpresent_<=0 && HFpresent_<=0)
	status_global_=-1;
      else
	status_global_=1;
      
      // Set other statuses based on subdetectors
      status_HO0_    = status_HO_;
      status_HO12_   = status_HO_;
      status_HFlumi_ = status_HF_;
      
      if (debug_>1)
	{
	  std::cout <<"Insufficient events processed.  Subdetector status is:"<<std::endl;
	  std::cout<<"\tHB: "<<status_HB_<<std::endl;
	  std::cout<<"\tHE: "<<status_HE_<<std::endl;
	  std::cout<<"\tHO: "<<status_HO_<<std::endl;
	  std::cout<<"\tHF: "<<status_HF_<<std::endl;
	  std::cout<<"\tHO0: "<<status_HO0_<<std::endl;
	  std::cout<<"\tHO12: "<<status_HO12_<<std::endl;
	  std::cout<<"\tHFlumi: "<<status_HFlumi_<<std::endl;
	}

      fillReportSummary(LS);
      return;
    }
  if (EnoughEvents_!=0) EnoughEvents_->setBinContent(clients_.size()+1,1); // summary is good to go!

  // set status to 0 if subdetector is present (or assumed present)
  if (HBpresent_>0) status_HB_=0;
  if (HEpresent_>0) status_HE_=0;
  if (HOpresent_>0) {status_HO_=0; status_HO0_=0; status_HO12_=0;}
  if (HFpresent_>0) {status_HF_=0; status_HFlumi_=0;}

  if (HBpresent_>0 || HEpresent_>0 ||
      HOpresent_>0 || HFpresent_>0 ) 
    status_global_=0;

  // don't want to fool with variable-sized arrays at the moment; revisit later
  //const unsigned int csize=clients_.size();
  double localHB[20]={0};
  double localHE[20]={0};
  double localHF[20]={0};
  double localHO[20]={0};
  double localHFlumi[20]={0};
  double localHO0[20]={0};
  double localHO12[20]={0};

  // reset all depth histograms
  if (SummaryMapByDepth==0)
    {
      if (debug_>0)
	std::cout <<"<HcalSummaryClient::analyze>  ERROR:  SummaryMapByDepth can't be found!"<<std::endl;
    }
  else 
    {
      for (unsigned int i=0;i<(SummaryMapByDepth->depth).size();++i)
	SummaryMapByDepth->depth[i]->Reset();
 
      int etabins=-9999;
      int phibins=-9999;
  
      // Get Channel Status histograms here
      std::vector<MonitorElement*> chStat;
      chStat.push_back(dqmStore_->get(prefixME_+"HcalInfo/ChannelStatus/HB HE HF Depth 1 ChannelStatus"));
      chStat.push_back(dqmStore_->get(prefixME_+"HcalInfo/ChannelStatus/HB HE HF Depth 2 ChannelStatus"));
      chStat.push_back(dqmStore_->get(prefixME_+"HcalInfo/ChannelStatus/HE Depth 3 ChannelStatus"));
      chStat.push_back(dqmStore_->get(prefixME_+"HcalInfo/ChannelStatus/HO Depth 4 ChannelStatus"));


      for (int d=0;d<4;++d)
	{
	  etabins=(SummaryMapByDepth->depth[d])->getNbinsX();
	  phibins=(SummaryMapByDepth->depth[d])->getNbinsY();
	  for (int eta=1;eta<=etabins;++eta)
	    {
	      int ieta=CalcIeta(eta-1,d+1);
	      for (int phi=1;phi<=phibins;++phi)
		{
		  // local phi counter is the same as iphi
		  // for |ieta|>20, iphi%2==0 cells are unphysical; skip 'em
		  // for |ieta|>39, iphi%4!=3 cells are unphysical
		  if (abs(ieta)>20 && phi%2==0) continue;
		  if (abs(ieta)>39 && phi%4!=3) continue;

		  // First loop calculates "local" error rates for each individual client
		  // This must be done separately from the SummaryMap overall loop, because that loop issues a
		  // 'break' the first time an error is found (to avoid double-counting multiple errors in a single channel).
		  for (unsigned int cl=0;cl<clients_.size();++cl)
		    {
		      if (clients_[cl]->ProblemCellsByDepth==0) continue;

		      if ((clients_[cl]->ProblemCellsByDepth)->depth[d]==0) continue;
		      if ((clients_[cl]->ProblemCellsByDepth)->depth[d]->getBinContent(eta,phi)>clients_[cl]->minerrorrate_)
			{
			  if (isHF(eta-1,d+1)) 
			    {
			      ++localHF[cl];
			      if ((d==0 && (abs(ieta)==33 || abs(ieta)==34)) ||   // depth 1, rings 33,34
				  (d==1 && (abs(ieta)==35 || abs(ieta)==36)))     // depth 2, rings 35,36
				++localHFlumi[cl]; 
			    }
			  else if (isHO(eta-1,d+1)) 
			    {
			      ++localHO[cl];
			      if (abs(ieta)<5) ++localHO0[cl]; 
			      else ++localHO12[cl]; 
			    }
			  else if (isHB(eta-1,d+1)) ++localHB[cl];
			  else if (isHE(eta-1,d+1)) ++localHE[cl];
			}
		    } // for (loop on clients_.size() to determine individual client error rates)

		  // Check for certification errors -- do we want to add some extra warnings (filling channel status db plot with new value, etc?) in this case?

		  if (UseBadChannelStatusInSummary_ && chStat[d]!=0)
		    {
		      double chanStat=chStat[d]->getBinContent(eta,phi);
		      // chanStat<0 indicates original status from database was <0; this is counted as an error,
		      // since such values should never appear in the database.
		      if (chanStat<0)
			{
			  if (isHF(eta-1,d+1))
			    {
			      ++status_HF_;
			      if ((d==0 && (abs(ieta)==33 || abs(ieta)==34)) ||   // depth 1, rings 33,34
			      (d==1 && (abs(ieta)==35 || abs(ieta)==36)))     // depth 2, rings 35,36
				{
			      ++status_HFlumi_; 
				}
			      continue; // don't bother looking at individual clients for results; channel status is already corrupted
			    }
			  else if (isHO(eta-1,d+1))
			    {
			      ++status_HO_;
			      if (abs(ieta)<5) 
				++status_HO0_; 
			      else ++status_HO12_; 
			      continue;
			    }
			  else if (isHB(eta-1,d+1))
			    {
			      ++status_HB_;
			      continue;
			    }
			  else if (isHE(eta-1,d+1))
			    {
			      ++status_HE_;
			      continue;
			    }
			} // if (chanStat<0)
		    } // if (UseBadChannelStatusInSummary_)

		  // loop over all client tests
		 
		  // SummaryMapByDepth is slightly different from previous version -- it now just shows cells
		  // that contribute as "problems", rather than giving good channels a status of 1, and bad a status of 0
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
			    SummaryMapByDepth->depth[d]->setBinContent(eta,phi,999); // known problems filled with a value of 999
			  if (isHF(eta-1,d+1)) 
			    {
			      ++status_HF_;
			      if ((d==0 && (abs(ieta)==33 || abs(ieta)==34)) ||   // depth 1, rings 33,34
				  (d==1 && (abs(ieta)==35 || abs(ieta)==36)))     // depth 2, rings 35,36
				{
				  ++status_HFlumi_; 
				}
			    }
			  else if (isHO(eta-1,d+1)) 
			    {
			      ++status_HO_;
			      if (abs(ieta)<5) 
				  ++status_HO0_; 
			      else ++status_HO12_; 
			    }
			  else if (isHB(eta-1,d+1)) ++status_HB_;
			  else if (isHE(eta-1,d+1)) ++status_HE_;
			  break; // man, this break causes problems for certificationMap!!! -- Jason;   WHY?  -- Jeff
			}
		    } // for (main loop on clients_.size() to calculate reportSummary statuses)
		}
	    }
	} // for (int d=0;d<4;++d)

      FillUnphysicalHEHFBins(*SummaryMapByDepth);
    } // else (SummaryMapByDepth exists)

  // We've checked all problems; now compute overall status
  int totalcells=0;
  std::map<std::string, int>::const_iterator it;

  if (HBpresent_>0)
    {
      status_global_+=status_HB_; 
      it=subdetCells_.find("HB");
      totalcells+=it->second;
      status_HB_= 1-(status_HB_/it->second);
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHB[i]=1-(1.*localHB[i]/it->second);
	  localHB[i]=std::max(0.,localHB[i]);
	}
      status_HB_=std::max(0.,status_HB_); // converts fraction of bad channels to good fraction
    }
  else status_HB_=-1; // enoughevents_ can be true even if HB not present; need to set status_HB_=-1 in both cases
 
  if (HEpresent_>0)
    {
      status_global_+=status_HE_;
      it=subdetCells_.find("HE");
      totalcells+=it->second;
      status_HE_= 1-(status_HE_/it->second);
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHE[i]=1-(1.*localHE[i]/it->second);
	  localHE[i]=std::max(0.,localHE[i]);
	}
      status_HE_=std::max(0.,status_HE_); // converts fraction of bad channels to good fraction
    }
  else status_HE_=-1;
 
  if (HOpresent_>0)
    {
      status_global_+=status_HO_;
      it=subdetCells_.find("HO");
      totalcells+=it->second;
      status_HO_= 1-(status_HO_/it->second);
      status_HO_=std::max(0.,status_HO_); // converts fraction of bad channels to good fraction
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHO[i]=1-(1.*localHO[i]/it->second);
	  localHO[i]=std::max(0.,localHO[i]);
	}
      it=subdetCells_.find("HO0");
      status_HO0_= 1-(status_HO0_/it->second);
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHO0[i]=1-(1.*localHO0[i]/it->second);
	  localHO0[i]=std::max(0.,localHO0[i]);
	}
      status_HO0_=std::max(0.,status_HO0_); // converts fraction of bad channels to good fraction
      it=subdetCells_.find("HO12");
      status_HO12_= 1-(status_HO12_/it->second);
      status_HO12_=std::max(0.,status_HO12_); // converts fraction of bad channels to good fraction
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHO12[i]=1-(1.*localHO12[i]/it->second);
	  localHO12[i]=std::max(0.,localHO12[i]);
	}
    }
  else
    {
      status_HO_=-1;
      status_HO0_=-1;
      status_HO12_=-1;
    }
  if (HFpresent_>0)
    {
      status_global_+=status_HF_;
      it=subdetCells_.find("HF");
      totalcells+=it->second;
      status_HF_= 1-(status_HF_/it->second);
      status_HF_=std::max(0.,status_HF_); // converts fraction of bad channels to good fraction
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHF[i]=1-(1.*localHF[i]/it->second);
	  localHF[i]=std::max(0.,localHF[i]);
	}
      it=subdetCells_.find("HFlumi");
      status_HFlumi_= 1-(status_HFlumi_/it->second);
      status_HFlumi_=std::max(0.,status_HFlumi_); // converts fraction of bad channels to good fraction
      for (unsigned int i=0;i<clients_.size();++i)
	{
	  localHFlumi[i]=1-(1.*localHFlumi[i]/it->second);
	  localHFlumi[i]=std::max(0.,localHFlumi[i]);
	}
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
      status_global_=std::max(0.,status_global_); // convert to good fraction
    }
 

  // Fill certification map here

  dqmStore_->setCurrentFolder(prefixME_+"HcalInfo");
  certificationMap_=dqmStore_->get(prefixME_+"HcalInfo/CertificationMap");
  if (certificationMap_) dqmStore_->removeElement(certificationMap_->getName());
  certificationMap_=dqmStore_->book2D("CertificationMap","Certification Map",7,0,7,
				      clients_.size()+1,0,clients_.size()+1);

  certificationMap_->getTH2F()->GetYaxis()->SetBinLabel(1,"Summary");
  (certificationMap_->getTH2F())->SetOption("textcolz");

  for (int i=0;i<(int)clients_.size();++i)
    {
      certificationMap_->getTH2F()->GetYaxis()->SetBinLabel(i+2,(clients_[i]->name()).c_str());
    }
  certificationMap_->getTH2F()->GetYaxis()->SetLabelSize(0.02);
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(1,"HB");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(2,"HE");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(3,"HO");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(4,"HF");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(5,"HO0");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(6,"HO12");
  certificationMap_->getTH2F()->GetXaxis()->SetBinLabel(7,"HFlumi");
  certificationMap_->getTH2F()->SetMinimum(-1);
  certificationMap_->getTH2F()->SetMaximum(1);

  for (unsigned int i=0;i<clients_.size();++i)
    {
      certificationMap_->setBinContent(1,i+2,localHB[i]);
      certificationMap_->setBinContent(2,i+2,localHE[i]);
      certificationMap_->setBinContent(3,i+2,localHO[i]);
      certificationMap_->setBinContent(4,i+2,localHF[i]);
      certificationMap_->setBinContent(5,i+2,localHO0[i]);
      certificationMap_->setBinContent(6,i+2,localHO12[i]);
      certificationMap_->setBinContent(7,i+2,localHF[i]);
    }
  certificationMap_->setBinContent(1,1,status_HB_);
  certificationMap_->setBinContent(2,1,status_HE_);
  certificationMap_->setBinContent(3,1,status_HO_);
  certificationMap_->setBinContent(4,1,status_HF_);
  certificationMap_->setBinContent(5,1,status_HO0_);
  certificationMap_->setBinContent(6,1,status_HO12_);
  certificationMap_->setBinContent(7,1,status_HF_);
  fillReportSummary(LS);
} // analyze

void HcalSummaryClient::fillReportSummary(int LS)
{

  // We've now checked all tasks; now let's calculate summary values
 
  if (debug_>2)  std::cout <<"<HcalSummaryClient::fillReportSummary>"<<std::endl;

  if (debug_>3) 
    {
      std::cout <<"STATUS = "<<std::endl;
      std::cout <<"HB = "<<status_HB_<<std::endl;
      std::cout <<"HE = "<<status_HE_<<std::endl;
      std::cout <<"HO = "<<status_HO_<<std::endl;
      std::cout <<"HF = "<<status_HF_<<std::endl;
      std::cout <<"HO0 = "<<status_HO0_<<std::endl;
      std::cout <<"HO12 = "<<status_HO12_<<std::endl;
      std::cout <<"HFlumi = "<<status_HFlumi_<<std::endl;
    }

  // put the summary values into MonitorElements 

  if (LS>0)
    {
      if (StatusVsLS_)
        {
	  StatusVsLS_->setBinContent(LS,1,status_HB_);
	  StatusVsLS_->setBinContent(LS,2,status_HE_);
	  StatusVsLS_->setBinContent(LS,3,status_HO_);
	  StatusVsLS_->setBinContent(LS,4,status_HF_);
	  StatusVsLS_->setBinContent(LS,5,status_HO0_);
	  StatusVsLS_->setBinContent(LS,6,status_HO12_);
	  StatusVsLS_->setBinContent(LS,7,status_HFlumi_);
	}
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
      // Set reportMap underflow bin based on whether enough total events have been processed
      if (enoughevents_==false)
	reportMap_->setBinContent(0,0,-1);
      else
	reportMap_->setBinContent(0,0,1);
    }
  else if (debug_>0) std::cout <<"<HcalSummaryClient::fillReportSummary> CANNOT GET REPORT SUMMARY MAP!!!!!"<<std::endl;

  if (reportMapShift_)
    {
      reportMapShift_->setBinContent(1,1,status_HB_);
      reportMapShift_->setBinContent(2,1,status_HE_);
      reportMapShift_->setBinContent(3,1,status_HO_);
      reportMapShift_->setBinContent(4,1,status_HF_);
      reportMapShift_->setBinContent(5,1,status_HO0_);
      reportMapShift_->setBinContent(6,1,status_HO12_);
      // Set reportMap underflow bin based on whether enough total events have been processed
      if (enoughevents_==false)
	reportMapShift_->setBinContent(0,0,-1);
      else
	reportMapShift_->setBinContent(0,0,1);
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
	  if (debug_>0) std::cout <<"<HcalSummaryClient::analyze()>  Could not get Monitor Element named 'Hcal_"<<subdets[i]<<"'"<<std::endl;
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


void HcalSummaryClient::fillReportSummaryLSbyLS(int LS)
{

  MonitorElement* me;
  dqmStore_->setCurrentFolder(prefixME_+"LSbyLS_Hcal/LSvalues");
  
  float status_HB=-1;
  float status_HE=-1;
  float status_HO=-1;
  float status_HF=-1;
  float status_HO0=-1;
  float status_HO12=-1;
  float status_HFlumi=-1;
  float status_global=-1;

  me=dqmStore_->get(prefixME_+"LSbyLS_Hcal/LSvalues/ProblemsThisLS");
  if (me!=0)
    {
      //check to see if enough events were processed to make tests
      int events=(int)me->getBinContent(-1);
      if (events>0)
	{
	  std::map<std::string, int>::const_iterator it;
	  int totalcells=0;

	  status_HB=me->getBinContent(1,1);
	  status_HE=me->getBinContent(2,1);
	  status_HO=me->getBinContent(3,1);
	  status_HF=me->getBinContent(4,1);
	  status_HO0=me->getBinContent(5,1);
	  status_HO12=me->getBinContent(6,1);
	  status_HFlumi=me->getBinContent(7,1);

	  status_global=status_HB+status_HE+status_HO+status_HF;
	  if (debug_>1) std::cout <<"<HcalSummaryClient::fillReportsummaryLSbyLS>   BAD CHANNELS*EVENTS = HB: "<<status_HB<<" HE: "<<status_HE<<" HO: "<<status_HO<<" HO0: "<<status_HO0<<" HO12: "<<status_HO12<<" HF:"<<status_HF<<" HFlumi: "<<status_HFlumi<<"  TOTAL BAD CHANNELS*EVENTS = "<<status_global<<"  TOTAL EVENTS = "<<events<<std::endl;

	  it=subdetCells_.find("HB");
	  totalcells+=it->second;
	  if (it->second>0)
	    status_HB=1-(status_HB)/events/it->second;

	  it=subdetCells_.find("HE");
	  totalcells+=it->second;
	  if (it->second>0)
	    status_HE=1-(status_HE)/events/it->second;

	  it=subdetCells_.find("HO");
	  totalcells+=it->second;
	  if (it->second>0)
	    status_HO=1-(status_HO)/events/it->second;

	  it=subdetCells_.find("HF");
	  totalcells+=it->second;
	  if (it->second>0)
	    status_HF=1-(status_HF)/events/it->second;

	  it=subdetCells_.find("HO0");
	  if (it->second>0)
	    status_HO0=1-(status_HO0)/events/it->second;

	  it=subdetCells_.find("HO12");
	  if (it->second>0)
	    status_HO12=1-(status_HO12)/events/it->second;

	  it=subdetCells_.find("HFlumi");
	  if (it->second>0)
	    status_HFlumi=1-(status_HFlumi)/events/it->second;
	  if (totalcells>0)
	    status_global=1-status_global/events/totalcells;
	  if (debug_>1) std::cout <<"<HcalSummaryClient::fillReportsummaryLSbyLS>   STATUS= HB: "<<status_HB<<" HE: "<<status_HE<<" HO: "<<status_HO<<" HO0: "<<status_HO0<<" HO12: "<<status_HO12<<" HF:"<<status_HF<<" HFlumi: "<<status_HFlumi<<"  GLOBAL STATUS = "<<status_global<<"  TOTAL EVENTS = "<<events<<std::endl;
	} // if (events(>0)
    } // if (me!=0)

  dqmStore_->setCurrentFolder(subdir_);
  if (reportMap_)
    {
      reportMap_->setBinContent(1,1,status_HB);
      reportMap_->setBinContent(2,1,status_HE);
      reportMap_->setBinContent(3,1,status_HO);
      reportMap_->setBinContent(4,1,status_HF);
      reportMap_->setBinContent(5,1,status_HO0);
      reportMap_->setBinContent(6,1,status_HO12);
      reportMap_->setBinContent(7,1,status_HFlumi);
      // Set reportMap underflow bin based on whether enough total events have been processed
      if (enoughevents_==false)
	reportMap_->setBinContent(0,0,-1);
      else
	reportMap_->setBinContent(0,0,1);

    }
  else if (debug_>0) std::cout <<"<HcalSummaryClient::fillReportSummaryLSbyLS> CANNOT GET REPORT SUMMARY MAP!!!!!"<<std::endl;

  if (reportMapShift_)
    {
      reportMapShift_->setBinContent(1,1,status_HB);
      reportMapShift_->setBinContent(2,1,status_HE);
      reportMapShift_->setBinContent(3,1,status_HO);
      reportMapShift_->setBinContent(4,1,status_HF);
      reportMapShift_->setBinContent(5,1,status_HO0);
      reportMapShift_->setBinContent(6,1,status_HO12);
      // Set reportMap underflow bin based on whether enough total events have been processed
      if (enoughevents_==false)
	reportMapShift_->setBinContent(0,0,-1);
      else
	reportMapShift_->setBinContent(0,0,1);

    }
  else if (debug_>0) std::cout <<"<HcalSummaryClient::fillReportSummaryLSbyLS> CANNOT GET REPORT SUMMARY MAP!!!!!"<<std::endl;

  me=dqmStore_->get(subdir_+"reportSummary");
  // Clear away old versions
  if (me) me->Fill(status_global);

  // Create floats for each subdetector status
  std::string subdets[7] = {"HB","HE","HO","HF","HO0","HO12","HFlumi"};
  for (unsigned int i=0;i<7;++i)
    {
      // Create floats showing subtasks status
      dqmStore_->setCurrentFolder( subdir_+ "reportSummaryContents" );  
      me=dqmStore_->get(subdir_+"reportSummaryContents/Hcal_"+subdets[i]);
      if (me==0)
	{
	  if (debug_>0) std::cout <<"<HcalSummaryClient::LSbyLS>  Could not get Monitor Element named 'Hcal_"<<subdets[i]<<"'"<<std::endl;
	  continue;
	}
      if (subdets[i]=="HB") me->Fill(status_HB);
      else if (subdets[i]=="HE") me->Fill(status_HE);
      else if (subdets[i]=="HO") me->Fill(status_HO);
      else if (subdets[i]=="HF") me->Fill(status_HF);
      else if (subdets[i]=="HO0") me->Fill(status_HO0);
      else if (subdets[i]=="HO12") me->Fill(status_HO12);
      else if (subdets[i]=="HFlumi") me->Fill(status_HFlumi);
    } // for (unsigned int i=0;...)

  
  if (StatusVsLS_)
    {
      StatusVsLS_->setBinContent(LS,1,status_HB);
      StatusVsLS_->setBinContent(LS,2,status_HE);
      StatusVsLS_->setBinContent(LS,3,status_HO);
      StatusVsLS_->setBinContent(LS,4,status_HF);
      StatusVsLS_->setBinContent(LS,5,status_HO0);
      StatusVsLS_->setBinContent(LS,6,status_HO12);
      StatusVsLS_->setBinContent(LS,7,status_HFlumi);
    }

  return;


} // void HcalSummaryClient::fillReportSummaryLSbyLS()



void HcalSummaryClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  // set total number of cells in each subdetector
  subdetCells_.insert(std::make_pair("HB",2592));
  subdetCells_.insert(std::make_pair("HE",2592));
  subdetCells_.insert(std::make_pair("HO",2160));
  subdetCells_.insert(std::make_pair("HF",1728));
  subdetCells_.insert(std::make_pair("HO0",576));
  subdetCells_.insert(std::make_pair("HO12",1584));
  subdetCells_.insert(std::make_pair("HFlumi",288));  // 8 rings, 36 cells/ring
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
    EnoughEvents_=dqmStore_->book1D("EnoughEvents","Enough Events Passed From Each Task To Form Summary",1+(int)clients_.size(),0,1+(int)clients_.size());
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    EnoughEvents_->setBinLabel(i+1,clients_[i]->name());
  EnoughEvents_->setBinLabel(1+(int)clients_.size(),"Summary");

  if (MinEvents_==0)
    MinEvents_=dqmStore_->book1D("MinEvents","Minimum Events Required From Each Task To Form Summary",
				 1+(int)clients_.size(),0,1+(int)clients_.size());
  int summin=0;
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    {
      MinEvents_->setBinLabel(i+1,clients_[i]->name());
      MinEvents_->setBinContent(i+1,clients_[i]->minevents_);
      summin=std::max(summin,clients_[i]->minevents_);
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

  // Extra fix provided by Giuseppe
  
  if (SummaryMapByDepth!=0)
    {
      delete SummaryMapByDepth;
      SummaryMapByDepth=0;
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
  //(reportMap_->getTH2F())->SetOption("textcolz");
  (reportMap_->getTH2F())->SetMinimum(-1);
  (reportMap_->getTH2F())->SetMaximum(1);

  if (reportMapShift_)
    dqmStore_->removeElement(reportMapShift_->getName());
  reportMapShift_ = dqmStore_->book2D("reportSummaryMapShift","reportSummaryMapShift",
				 6,0,6,1,0,1);
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(1,"HB");
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(2,"HE");
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(3,"HO");
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(4,"HF");
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(5,"HO0");
  (reportMapShift_->getTH2F())->GetXaxis()->SetBinLabel(6,"HO12");
  (reportMapShift_->getTH2F())->GetYaxis()->SetBinLabel(1,"Status");
  (reportMapShift_->getTH2F())->SetMarkerSize(3);
  (reportMapShift_->getTH2F())->SetOption("text90colz");
  //(reportMapShift_->getTH2F())->SetOption("textcolz");
  (reportMapShift_->getTH2F())->SetMinimum(-1);
  (reportMapShift_->getTH2F())->SetMaximum(1);

  // Set initial counters to -1 (unknown)
  status_global_=-1; 
  status_HB_=-1; 
  status_HE_=-1; 
  status_HO_=-1; 
  status_HF_=-1; 

  status_HO0_=-1;
  status_HO12_=-1;
  status_HFlumi_=-1;
  for (int i=1;i<=(reportMap_->getTH2F())->GetNbinsX();++i)
    reportMap_->setBinContent(i,1,-1);
  for (int i=1;i<=(reportMapShift_->getTH2F())->GetNbinsX();++i)
    reportMapShift_->setBinContent(i,1,-1);
} // void HcalSummaryClient::beginRun(void)


void HcalSummaryClient::endRun(void){}

void HcalSummaryClient::setup(void){}
void HcalSummaryClient::cleanup(void){}

bool HcalSummaryClient::hasErrors_Temp(void){  return false;}

bool HcalSummaryClient::hasWarnings_Temp(void){return false;}
bool HcalSummaryClient::hasOther_Temp(void){return false;}
bool HcalSummaryClient::test_enabled(void){return true;}

void HcalSummaryClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual){return;}


HcalSummaryClient::~HcalSummaryClient()
{}
