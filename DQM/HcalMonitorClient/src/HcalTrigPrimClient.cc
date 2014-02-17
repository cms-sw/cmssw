#include "DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalTrigPrimClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.22 $
 * \author J. Temple
 * \brief Hcal Trigger Primitive Client class
 */

HcalTrigPrimClient::HcalTrigPrimClient(std::string myname)
{
  name_=myname;
}

HcalTrigPrimClient::HcalTrigPrimClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TrigPrimFolder","TrigPrimMonitor_Hcal/"); // TrigPrimMonitor
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("TrigPrim_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("TrigPrim_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  // Set error rate to 1%, not (earlier) value of 0.1% -- Jeff, 11 May 2010
  minerrorrate_ = ps.getUntrackedParameter<double>("TrigPrim_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.01));
  minevents_    = ps.getUntrackedParameter<int>("TrigPrim_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
}

void HcalTrigPrimClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalTrigPrimClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalTrigPrimClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalTrigPrimClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  double totalevents=0;
  int etabins=0, phibins=0;
  double problemvalue=0;
  enoughevents_=false;  // assume we lack sufficient events until proven otherwise

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
    }
  for  (unsigned int d=0;d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]!=0) 
	{
	  ProblemCellsByDepth->depth[d]->Reset();
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMinimum(0.);
	}
    }

  for  (unsigned int d=0;d<ProblemsByDepthZS_->depth.size();++d)
    {
      if (ProblemsByDepthZS_->depth[d]!=0) 
	{
	  ProblemsByDepthZS_->depth[d]->Reset();
	  (ProblemsByDepthZS_->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemsByDepthZS_->depth[d]->getTH2F())->SetMinimum(0.);
	}
    }

  for  (unsigned int d=0;d<ProblemsByDepthNZS_->depth.size();++d)
    {
      if (ProblemsByDepthNZS_->depth[d]!=0) 
	{
	  ProblemsByDepthNZS_->depth[d]->Reset();
	  (ProblemsByDepthNZS_->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemsByDepthNZS_->depth[d]->getTH2F())->SetMinimum(0.);
	}
    }

  // Get histograms that are used in testing
  // currently none used,

  std::vector<std::string> name = HcalEtaPhiHistNames();

  /*
    // This is a sample of how to get a histogram from the task that can then be used for evaluation purposes
  */
  MonitorElement* me;
  TH2F *goodZS=0;
  TH2F *badZS=0;
  TH2F* goodNZS=0;
  TH2F* badNZS=0;

  me=dqmStore_->get(subdir_+"Good TPs_ZS");
  if (!me && debug_>0)
    std::cout <<"<HcalTrigPrimClient::calculateProblems>  Could not get histogram named '"<<subdir_<<"Good TPs_ZS'"<<std::endl;
  else goodZS = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, goodZS, debug_);

  me=dqmStore_->get(subdir_+"Bad TPs_ZS");
  if (!me && debug_>0)
    std::cout <<"<HcalTrigPrimClient::calculateProblems>  Could not get histogram named '"<<subdir_<<"Bad TPs_ZS'"<<std::endl;
  else badZS = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, badZS, debug_);

  me=dqmStore_->get(subdir_+"noZS/Good TPs_noZS");
  if (!me && debug_>0)
    std::cout <<"<HcalTrigPrimClient::calculateProblems>  Could not get histogram named '"<<subdir_<<"noZS/Good TPs_noZS'"<<std::endl;
  else goodNZS = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, goodNZS, debug_);

  me=dqmStore_->get(subdir_+"noZS/Bad TPs_noZS");
  if (!me && debug_>0)
    std::cout <<"<HcalTrigPrimClient::calculateProblems>  Could not get histogram named '"<<subdir_<<"noZS/Bad TPs_noZS'"<<std::endl;
  else badNZS = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, badNZS, debug_);

  // get bin info from good histograms
  if (goodZS!=0)
    {
      etabins=goodZS->GetNbinsX();
      phibins=goodZS->GetNbinsY();
      totalevents=goodNZS->GetBinContent(0);
    }
  else if (goodNZS!=0)
    {
      etabins=goodNZS->GetNbinsX();
      phibins=goodNZS->GetNbinsY();
      totalevents=goodNZS->GetBinContent(0);
    }

  if (totalevents<minevents_) 
    {
      enoughevents_=false;
      if (debug_>2) std::cout <<"<HcalTrigPrimClient::calculateProblems()>  Not enough events!  events = "<<totalevents<<"  minimum required = "<<minevents_<<std::endl;
      return;
    }
  enoughevents_=true;

  // got good and bad histograms; now let's loop over them

  int ieta=-99, iphi=-99;
  int badvalZS=0, goodvalZS=0;
  int badvalNZS=0, goodvalNZS=0;
  for (int eta=1;eta<=etabins;++eta)
    {
      ieta=eta-33; // Patrick's eta-phi maps starts at ieta=-32
      for (int phi=1;phi<=phibins;++phi)
	{
	  badvalZS=0, goodvalZS=0;
	  badvalNZS=0, goodvalNZS=0;
	  iphi=phi;
	  if (badZS!=0) badvalZS=(int)badZS->GetBinContent(eta,phi);
	  if (badNZS!=0) badvalNZS=(int)badNZS->GetBinContent(eta,phi);
	  if (badvalZS+badvalNZS==0) continue;
	  if (goodZS!=0) goodvalZS=(int)goodZS->GetBinContent(eta,phi);
	  if (goodNZS!=0) goodvalNZS=(int)goodNZS->GetBinContent(eta,phi);

	  if (badvalNZS>0)
	    {
	      problemvalue=badvalNZS*1./(badvalNZS+goodvalNZS);
	      if (abs(ieta)<29) 
		{
		  ProblemsByDepthNZS_->depth[0]->Fill(ieta,iphi,problemvalue);
		  if (abs(ieta)==28) // TP 28 spans towers 28 and 29
		    ProblemsByDepthNZS_->depth[0]->Fill(ieta+abs(ieta)/ieta,iphi,problemvalue);
		}
	      else // HF
		{
		  /* HF TPs:
		     TP29 = ieta 29-31
		     TP30 = ieta 32-34
		     TP31 = ieta 35-37
		     TP38 = ieta 38-41
		     iphi = 1, 5, ..., with 1 covering iphi=1 and iphi=71, etc.
		  */
		  int newieta=-99;
		  for (int i=0;i<3;++i)
		    {
		      newieta=i+29+3*(abs(ieta)-29)+1; // shift values by 1 for HF in EtaPhiHistsplot
		      if (ieta<0) newieta*=-1;
		      ProblemsByDepthNZS_->depth[0]->Fill(newieta,iphi,problemvalue);
		      ProblemsByDepthNZS_->depth[0]->Fill(newieta,(iphi-2+72)%72,problemvalue);
		    }
		  if (abs(ieta)==32)
		    {
		      ProblemsByDepthNZS_->depth[0]->Fill(42*abs(ieta)/ieta,iphi,problemvalue);
		      ProblemsByDepthNZS_->depth[0]->Fill(newieta,(iphi-2+72)%72,problemvalue);
		    }
		}
	    } // errors found in NZS;
	  if (badvalZS>0)
	    {
	      problemvalue=badvalZS*1./(badvalZS+goodvalZS);
	      if (abs(ieta)<29) // Make special case for ieta=16 (HB/HE overlap?)
		{
		  ProblemsByDepthZS_->depth[0]->Fill(ieta,iphi,problemvalue);
		  if (abs(ieta)==28) // TP 28 spans towers 28 and 29
		    ProblemsByDepthZS_->depth[0]->Fill(ieta+abs(ieta)/ieta,iphi,problemvalue);
		}
	      else
		{
		  int newieta=-99;
		  for (int i=0;i<3;++i)
			{
			  newieta=i+29+3*(abs(ieta)-29)+1; // shift values by 1 for HF in EtaPhiHistsplot
			  if (ieta<0) newieta*=-1;
			  ProblemsByDepthZS_->depth[0]->Fill(newieta,iphi,problemvalue);
			  ProblemsByDepthZS_->depth[0]->Fill(newieta,(iphi-2+72)%72,problemvalue);
			}
		  if (abs(ieta)==32)
		    {
		      ProblemsByDepthZS_->depth[0]->Fill(42*abs(ieta)/ieta,iphi,problemvalue);
		      ProblemsByDepthZS_->depth[0]->Fill(42*abs(ieta)/ieta,(iphi-2+72)%72,problemvalue);
		    }
		}
	    } // errors found in ZS
	  if (badvalZS>0 || badvalNZS>0)
	    {
	      // Fill overall problem histograms with sum from both ZS & NZS, or ZS only?
	      //problemvalue=(badvalZS+badvalNZS)*1./(badvalZS+badvalNZS+goodvalZS+goodvalNZS);
	      
	      // Update on 8 March -- NZS shows lots of errors; let's not include that in problem cells just yet
	      if (badvalZS==0) continue;
	      problemvalue=(badvalZS*1.)/(badvalZS+goodvalZS);
	      if (abs(ieta)<29) // Make special case for ieta=16 (HB/HE overlap?)
		{
		  ProblemCellsByDepth->depth[0]->Fill(ieta,iphi,problemvalue);
		  ProblemCells->Fill(ieta,iphi,problemvalue);
		  if (abs(ieta)==28) // TP 28 spans towers 28 and 29
		    {
		      ProblemCellsByDepth->depth[0]->Fill(ieta+abs(ieta)/ieta,iphi,problemvalue);
		      ProblemCells->Fill(ieta+abs(ieta)/ieta,iphi,problemvalue);
		    }
		}
	      else
		{
		  int newieta=-99;
		  int newiphi=(iphi-2+72)%72;  // forward triggers combine two HF cells; *subtract* 2 to the iphi, and allow wraparound
		  // FIXME:
		  // iphi seems to start at 1 in Patrick's plots, continues mod 4;
		  // adjust in far-forward region, where cells start at iphi=3?  Check with Patrick.
		  for (int i=0;i<3;++i)
			{
			  newieta=i+29+3*(abs(ieta)-29)+1; // shift values by 1 for HF in EtaPhiHistsplot
			  if (ieta<0) newieta*=-1;
			  ProblemCellsByDepth->depth[0]->Fill(newieta,iphi,problemvalue);
			  ProblemCells->Fill(newieta,iphi,problemvalue);
			  ProblemCellsByDepth->depth[0]->Fill(newieta,newiphi,problemvalue);
			  ProblemCells->Fill(newieta,newiphi,problemvalue);
			}
		  if (abs(ieta)==32)
		    {
		      ProblemCellsByDepth->depth[0]->Fill(42*abs(ieta)/ieta,iphi,problemvalue);
		      ProblemCells->Fill(42*abs(ieta)/ieta,iphi,problemvalue);
		      ProblemCellsByDepth->depth[0]->Fill(42*abs(ieta)/ieta,newiphi,problemvalue);
		      ProblemCells->Fill(42*abs(ieta)/ieta,newiphi,problemvalue);
		    }
		}
	    }
	}
    } // for (int eta=1;eta<etabins;++eta)
    

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalTrigPrimClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
      return;
    }

  // Normalization of ProblemCell plot, in the case where there are errors in multiple depths
  etabins=(ProblemCells->getTH2F())->GetNbinsX();
  phibins=(ProblemCells->getTH2F())->GetNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	{
	  if (ProblemCells->getBinContent(eta+1,phi+1)>1. && ProblemCells->getBinContent(eta+1,phi+1)<999)
	    ProblemCells->setBinContent(eta+1,phi+1,1.);
	}
    }

  FillUnphysicalHEHFBins(*ProblemCellsByDepth);
  FillUnphysicalHEHFBins(*ProblemsByDepthZS_);
  FillUnphysicalHEHFBins(*ProblemsByDepthNZS_);
  FillUnphysicalHEHFBins(ProblemCells);
  return;
}

void HcalTrigPrimClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalTrigPrimClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalTrigPrimClient::endJob(){}

void HcalTrigPrimClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalTrigPrimClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemTriggerPrimitives",
				 " Problem Trigger Primitive Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_triggerprimitives");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem Trigger Primitive Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;

  dqmStore_->setCurrentFolder(subdir_+"problem_ZS");
  ProblemsByDepthZS_  = new EtaPhiHists();
  ProblemsByDepthZS_->setup(dqmStore_,"ZS Problem Trigger Primitive Rate");
  dqmStore_->setCurrentFolder(subdir_+"problem_NZS");
  ProblemsByDepthNZS_ = new EtaPhiHists();
  ProblemsByDepthNZS_->setup(dqmStore_,"NZS Problem Trigger Primitive Rate");
}

void HcalTrigPrimClient::endRun(void){analyze();}

void HcalTrigPrimClient::setup(void){}
void HcalTrigPrimClient::cleanup(void){}

bool HcalTrigPrimClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalTrigPrimClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
      return false;
    }
  int problemcount=0;
  int ieta=-9999;

  for (int depth=0;depth<4; ++depth)
    {
      int etabins  = (ProblemCells->getTH2F())->GetNbinsX();
      int phibins  = (ProblemCells->getTH2F())->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
          for (int hist_phi=0; hist_phi<phibins;++hist_phi)
            {
              ieta=CalcIeta(hist_eta,depth+1);
	      if (ieta==-9999) continue;
	      if (ProblemCellsByDepth->depth[depth]==0)
		  continue;
	      if (ProblemCellsByDepth->depth[depth]->getBinContent(hist_eta,hist_phi)>minerrorrate_)
		++problemcount;

	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;
}

bool HcalTrigPrimClient::hasWarnings_Temp(void){return false;}
bool HcalTrigPrimClient::hasOther_Temp(void){return false;}
bool HcalTrigPrimClient::test_enabled(void){return true;}


void HcalTrigPrimClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

} //void HcalTrigPrimClient::updateChannelStatus

HcalTrigPrimClient::~HcalTrigPrimClient()
{}
