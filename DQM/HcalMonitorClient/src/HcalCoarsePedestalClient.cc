#include "DQM/HcalMonitorClient/interface/HcalCoarsePedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalCoarsePedestalClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.6 $
 * \author J. Temple
 * \brief CoarsePedestalClient class
 */

HcalCoarsePedestalClient::HcalCoarsePedestalClient(std::string myname)
{
  name_=myname;
}

HcalCoarsePedestalClient::HcalCoarsePedestalClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("CoarsePedestalFolder","CoarsePedestalMonitor_Hcal/"); // CoarsePedestalMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("CoarsePedestal_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("CoarsePedestal_BadChannelStatusMask",
                                                          ps.getUntrackedParameter<int>("BadChannelStatusMask",
											((1<<HcalChannelStatus::HcalCellDead)|(1<<HcalChannelStatus::HcalCellHot)))); // identify channel status values to mask
  
  minerrorrate_ = ps.getUntrackedParameter<double>("CoarsePedestal_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.05));


  // minevents_ canbe overwritten by monitor task value
  minevents_    = ps.getUntrackedParameter<int>("CoarsePedestal_minevents",
						ps.getUntrackedParameter<int>("minevents",1));

  ProblemCellsByDepth=0;

}

void HcalCoarsePedestalClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalCoarsePedestalClient::analyze()"<<std::endl;
  calculateProblems();
} // void HcalCoarsePedestalClient::analyze()

void HcalCoarsePedestalClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalCoarsePedestalClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  //int totalevents=0; // events checked on a channel-by-channel basis
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;

  if (CoarsePedDiff!=0) CoarsePedDiff->Reset();

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
      (ProblemCells->getTH2F())->SetOption("colz");
    }
  for  (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]!=0) 
	{
	  ProblemCellsByDepth->depth[d]->Reset();
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMinimum(0.);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetOption("colz");
	}
    }

  // Get histograms that are used in testing
  TH2F* CoarsePedestalsSumByDepth[4];
  TH2F* CoarsePedestalsOccByDepth[4];

  std::vector<std::string> name = HcalEtaPhiHistNames();
  bool gothistos=true;

  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      std::string s=subdir_+"CoarsePedestalSumPlots/"+name[i]+"Coarse Pedestal Summed Map";
      me=dqmStore_->get(s.c_str());
      if (me==0) 
	{
	  gothistos=false;
	  if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::calculateProblems> Could not get histogram with name "<<s<<std::endl;
	}
      CoarsePedestalsSumByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, CoarsePedestalsSumByDepth[i], debug_);

      s=subdir_+"CoarsePedestalSumPlots/"+name[i]+"Coarse Pedestal Occupancy Map";
      me=dqmStore_->get(s.c_str());
      if (me==0) 
	{
	  gothistos=false;
	  if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::calculateProblems> Could not get histogram with name "<<s<<std::endl;
	}
      CoarsePedestalsOccByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, CoarsePedestalsOccByDepth[i], debug_);

    } // for (int i=0;i<4;++i)

  if (gothistos==false)
    {
      if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::calculateProblems> Unable to get all necessary histograms to evaluate problem rate"<<std::endl;
      return;
    }

  enoughevents_=true;  // Always set this to true, so that pedestal monitoring doesn't hold up 

  int numevents=0;
  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;
      if (CoarsePedestalsSumByDepth[d]==0 || 
	  CoarsePedestalsOccByDepth[d]==0 ||
	  DatabasePedestalsADCByDepth[d]==0) continue;

      if (CoarsePedestalsByDepth->depth[d]!=0)
	CoarsePedestalsByDepth->depth[d]->Reset();

      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      if (abs(ieta)>20 && (phi+1)%2==0)
		continue;
	      if (abs(ieta)>39 && (phi+1)%4!=3)
		continue;
	      numevents=(int)CoarsePedestalsOccByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (numevents==0 || numevents<minevents_)
		{
		  if (debug_>1)
		    std::cout <<"NOT ENOUGH EVENTS for channel ("<<ieta<<", "<<phi+1<<", "<<d+1<<")  numevents = "<<numevents<<"  minevents = "<<minevents_<<std::endl;
		  continue;  // insufficient pedestal information available for this channel; continue on?
		}

	      problemvalue=1.*CoarsePedestalsSumByDepth[d]->GetBinContent(eta+1,phi+1)/numevents;
	      CoarsePedestalsByDepth->depth[d]->setBinContent(eta+1,phi+1,problemvalue);
	      problemvalue=(problemvalue-DatabasePedestalsADCByDepth[d]->GetBinContent(eta+1,phi+1));
	      CoarsePedDiff->Fill(problemvalue);
	      problemvalue=fabs(problemvalue);
	      // Pedestal status is cumulative (DetDiag pedestals perform resets after each calibration cycle, and save each output)
	      // Either a channels pedestal is 'bad' (outside of allowed range) or 'good' (in range)
	      problemvalue>ADCDiffThresh_ ? problemvalue=1 : problemvalue=0;
	      if (debug_>0 && problemvalue==1)
		std::cout <<"<HcalCoarsePedestalClient> Problem found for channel ("<<ieta<<", "<<phi+1<<", "<<d+1<<")"<<std::endl;
	      zside=0;
	      if (isHF(eta,d+1)) // shift ieta by 1 for HF
		ieta<0 ? zside = -1 : zside = 1;
	      // For problem cells that exceed our allowed rate,
	      // set the values to -1 if the cells are already marked in the status database
	      if (problemvalue>minerrorrate_)
		{
		  HcalSubdetector subdet=HcalEmpty;
		  if (isHB(eta,d+1))subdet=HcalBarrel;
		  else if (isHE(eta,d+1)) subdet=HcalEndcap;
		  else if (isHF(eta,d+1)) subdet=HcalForward;
		  else if (isHO(eta,d+1)) subdet=HcalOuter;
		  HcalDetId hcalid(subdet, ieta, phi+1, (int)(d+1));
		  if (badstatusmap.find(hcalid)!=badstatusmap.end())
		    problemvalue=999;
		}
	      ProblemCellsByDepth->depth[d]->setBinContent(eta+1,phi+1,problemvalue);
	      if (ProblemCells!=0) ProblemCells->Fill(ieta+zside,phi+1,problemvalue);
	    } // loop on phi
	} // loop on eta
    } // loop on depth

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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
  FillUnphysicalHEHFBins(ProblemCells);
  FillUnphysicalHEHFBins(*CoarsePedestalsByDepth);
  return;
}


void HcalCoarsePedestalClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalCoarsePedestalClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}

void HcalCoarsePedestalClient::endJob(){}

void HcalCoarsePedestalClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  CoarsePedestalsByDepth = new EtaPhiHists();
  CoarsePedestalsByDepth->setup(dqmStore_," Coarse Pedestal Map");

  CoarsePedDiff=dqmStore_->book1D("PedRefDiff","(Pedestal-Reference)",200,-10,10);

  problemnames_.clear();
  ProblemCells=dqmStore_->book2D(" ProblemCoarsePedestals",
				 " Problem Coarse Pedestal Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_coarsepedestals");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem Coarse Pedestal Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;
  
  std::vector<std::string> name = HcalEtaPhiHistNames();
  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      std::string s=prefixME_+"HcalInfo/PedestalsFromCondDB/"+name[i]+"ADC Pedestals From Conditions DB";
      me=dqmStore_->get(s.c_str());
      if (me==0) 
	{
	  if (debug_>0) std::cout <<"<HcalCoarsePedestalClient::beginRun> Could not get histogram with name "<<s<<std::endl;
	}
      DatabasePedestalsADCByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, DatabasePedestalsADCByDepth[i], debug_);
    }
  std::string s=subdir_+"CoarsePedestal_parameters/ADCdiff_Problem_Threshold";
  me=dqmStore_->get(s.c_str());
  if (me==0)
    {
      if (debug_>0) 
	std::cout <<"<HcalCoarsePedestalClient::beginRun> Could not get value with name "<<s<<std::endl;
    }
  else 
    ADCDiffThresh_=me->getFloatValue();
  s=subdir_+"CoarsePedestal_parameters/minEventsNeededForPedestalCalculation";

  me=dqmStore_->get(s.c_str());
  int temp = 0;
  if (me==0) 
    {
      if (debug_>0)
	{
	  std::cout <<"<HcalCoarsePedestalClient::beginRun> Could not get value with name "<<s<<"\n\t  Continuing on using default 'minevents' value of "<<minevents_<<std::endl;
	}
    }
  else 
    temp=me->getIntValue();
  if (temp>minevents_)
    {
      if (debug_>0)
	std::cout <<"<HcalCoarsePedestalClient::beginRun>  Specified client 'minevents' value of "<<minevents_<<"  is less than minimum task 'minevents' value of "<<temp<<"\n\t  Setting client 'minevents' to "<<temp<<std::endl;
      minevents_=temp;
    }
} // void HcalCoarsePedestalClient::beginRun()

void HcalCoarsePedestalClient::endRun(void){analyze();}

void HcalCoarsePedestalClient::setup(void){}
void HcalCoarsePedestalClient::cleanup(void){}

bool HcalCoarsePedestalClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalCoarsePedestalClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalCoarsePedestalClient::hasWarnings_Temp(void){return false;}
bool HcalCoarsePedestalClient::hasOther_Temp(void){return false;}
bool HcalCoarsePedestalClient::test_enabled(void){return true;}


void HcalCoarsePedestalClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // client does not alter channel status yet;
  // look at dead cell or hot cell clients for example code
} //void HcalCoarsePedestalClient::updateChannelStatus

HcalCoarsePedestalClient::~HcalCoarsePedestalClient()
{}
