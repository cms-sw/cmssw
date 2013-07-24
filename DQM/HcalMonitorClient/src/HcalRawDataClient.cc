#include "DQM/HcalMonitorClient/interface/HcalRawDataClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <iostream>

/*
 * \file HcalRawDataClient.cc
 * 
 * $Date: 2012/07/04 15:43:18 $
 * $Revision: 1.14 $
 * \author J. St. John
 * \brief Hcal Raw Data Client class
 */

HcalRawDataClient::HcalRawDataClient(std::string myname)
{
  name_=myname;
}

HcalRawDataClient::HcalRawDataClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("RawDataFolder","RawDataMonitor_Hcal/"); // RawDataMonitor
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("RawData_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("RawData_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  
  minerrorrate_ = ps.getUntrackedParameter<double>("RawData_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.01));
  minevents_    = ps.getUntrackedParameter<int>("RawData_minevents",
						ps.getUntrackedParameter<int>("minevents",1));

  excludeHORing2_       = ps.getUntrackedParameter<bool>("excludeHOring2_backup",false);
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
}

void HcalRawDataClient::endLuminosityBlock() {
//  if (LBprocessed_==true) return;  // LB already processed
//  UpdateMEs();
//  LBprocessed_=true; 
  if (debug_>2) std::cout <<"\tHcalRawDataClient::endLuminosityBlock()"<<std::endl;
  calculateProblems();
  return;
}


void HcalRawDataClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalRawDataClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalRawDataClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalRawDataClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  double totalevents=0;
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;
  
  //Get number of events to normalize by
  MonitorElement* me;
  me = dqmStore_->get(subdir_+"Events_Processed_Task_Histogram");
  if (me) totalevents=me->getBinContent(1);

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
    }
  for  (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]!=0) 
	{
	  ProblemCellsByDepth->depth[d]->Reset();
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMinimum(0.);
	}
      for (unsigned int eta=0; eta<85;++eta) //spans largest ieta breadth
	{
	  for (unsigned int phi=0;phi<72;++phi) //spans largest (only!) iphi breadth
	    {
	      problemcount[eta][phi][d]=0.0;
	    }
	}
    }
  enoughevents_=true;

  // Try to read excludeHOring2 status from file
  
  MonitorElement* temp_exclude = dqmStore_->get(subdir_+"ExcludeHOring2");

  // If value can't be read from file, keep the excludeHOring2_backup status
  if (temp_exclude != 0)
    {
      if (temp_exclude->getIntValue()>0)
	excludeHORing2_ = true;  
      else
	excludeHORing2_ = false;
    }



  //Get the plots showing raw data errors,
  //fill problemcount[][][] 
  fillProblemCountArray();

  std::vector<std::string> name = HcalEtaPhiHistNames();

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;
    
      if (totalevents==0 || totalevents<minevents_) continue;
      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      problemvalue=0;
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;
	      problemvalue=((uint64_t) problemcount[eta][phi][d] );

	      if (problemvalue==0) continue;
	      problemvalue/=totalevents; // problem value is a rate; should be between 0 and 1
	      problemvalue = std::min(1.,problemvalue);
	      
	      zside=0;
	      if (isHF(eta,d+1)) // shift ieta by 1 for HF
		ieta<0 ? zside = -1 : zside = 1;
	      
	      if (debug_>0) std::cout <<"problemvalue = "<<problemvalue<<"  ieta = "<<zside<<"  iphi = "<<phi+1<<"  d = "<<d+1<<std::endl;
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
      if (debug_>0) std::cout <<"<HcalRawDataClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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
  return;
}

void HcalRawDataClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalRawDataClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalRawDataClient::endJob(){}

void HcalRawDataClient::stashHDI(int thehash, HcalDetId thehcaldetid) {
  //Let's not allow indexing off the array...
  if ((thehash<0)||(thehash>=(NUMDCCS*NUMSPGS*HTRCHANMAX)))return;
  //...but still do the job requested.
  hashedHcalDetId_[thehash] = thehcaldetid;
}


void HcalRawDataClient::beginRun(void)
{
  if (debug_>2) std::cout <<"<HcalRawDataClient::beginRun>"<<std::endl;
  edm::ESHandle<HcalDbService> pSetup;
  c->get<HcalDbRecord>().get( pSetup );

  if (debug_>2) std::cout <<"\t<HcalRawDataClient::beginRun> Get Hcal mapping"<<std::endl;
  readoutMap_=pSetup->getHcalMapping();
  DetId detid_;
  HcalDetId hcaldetid_; 

  // Build a map of readout hardware unit to calorimeter channel
  std::vector <HcalElectronicsId> AllElIds = readoutMap_->allElectronicsIdPrecision();
  uint32_t itsdcc    =0;
  uint32_t itsspigot =0;
  uint32_t itshtrchan=0;
  
  if (debug_>2) std::cout <<"\t<HcalRawDataClient::beginRun> Loop over AllEIds"<<std::endl;
  // by looping over all precision (non-trigger) items.
  for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin();
       eid != AllElIds.end();
       eid++) {

    //Get the HcalDetId from the HcalElectronicsId
    detid_ = readoutMap_->lookup(*eid);
    // NULL if illegal; ignore
    if (!detid_.null()) {
      if (detid_.det()!=4) continue; //not Hcal
      if (detid_.subdetId()!=HcalBarrel &&
	  detid_.subdetId()!=HcalEndcap &&
	  detid_.subdetId()!=HcalOuter  &&
	  detid_.subdetId()!=HcalForward) continue;

      itsdcc    =(uint32_t) eid->dccid(); 
      itsspigot =(uint32_t) eid->spigot();
      itshtrchan=(uint32_t) eid->htrChanId();
      hcaldetid_ = HcalDetId(detid_);
      stashHDI(hashup(itsdcc,itsspigot,itshtrchan),
	       hcaldetid_);
    } // if (!detid_.null()) 
  } 

  if (debug_>2) std::cout <<"\t<HcalRawDataClient::beginRun> Completed loop."<<std::endl;

  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalRawDataClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }

  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();
  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemRawData",
				 " Problem Raw Data Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_rawdata");
  ProblemCellsByDepth = new EtaPhiHists();

  ProblemCells->getTH2F()->SetMinimum(0);
  ProblemCells->getTH2F()->SetMaximum(1.05);

  ProblemCellsByDepth->setup(dqmStore_," Problem Raw Data Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());

  nevts_=0;
}

void HcalRawDataClient::endRun(void){analyze();}

void HcalRawDataClient::setup(void){}
void HcalRawDataClient::cleanup(void){}

bool HcalRawDataClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalRawDataClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalRawDataClient::hasWarnings_Temp(void){return false;}
bool HcalRawDataClient::hasOther_Temp(void){return false;}
bool HcalRawDataClient::test_enabled(void){return true;}


void HcalRawDataClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // see dead or hot cell code for an example

} //void HcalRawDataClient::updateChannelStatus


void HcalRawDataClient::getHardwareSpaceHistos(void){
  MonitorElement* me;
  std::string s;
  if (debug_>1) std::cout<<"\t<HcalRawDataClient>: getHardwareSpaceHistos()"<<std::endl;
  s=subdir_+"Corruption/01 Common Data Format violations";
  me=dqmStore_->get(s.c_str());  
  meCDFErrorFound_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, meCDFErrorFound_, debug_);
  if (!meCDFErrorFound_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/02 DCC Event Format violation";
  me=dqmStore_->get(s.c_str());  
  meDCCEventFormatError_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, meDCCEventFormatError_, debug_);
  if (!meDCCEventFormatError_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/03 OrN Inconsistent - HTR vs DCC";
  me=dqmStore_->get(s.c_str());  
  meOrNSynch_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, meOrNSynch_, debug_);
  if (!meOrNSynch_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/05 BCN Inconsistent - HTR vs DCC";
  me=dqmStore_->get(s.c_str());  
  meBCNSynch_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, meBCNSynch_, debug_);
  if (!meBCNSynch_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/06 EvN Inconsistent - HTR vs DCC";
  me=dqmStore_->get(s.c_str());  
  meEvtNumberSynch_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, meEvtNumberSynch_, debug_);
  if (!meEvtNumberSynch_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/07 LRB Data Corruption Indicators";
  me=dqmStore_->get(s.c_str());  
  LRBDataCorruptionIndicators_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, LRBDataCorruptionIndicators_, debug_);
  if (!LRBDataCorruptionIndicators_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/08 Half-HTR Data Corruption Indicators";
  me=dqmStore_->get(s.c_str());  
  HalfHTRDataCorruptionIndicators_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HalfHTRDataCorruptionIndicators_, debug_);
  if (!HalfHTRDataCorruptionIndicators_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;

  s=subdir_+"Corruption/09 Channel Integrity Summarized by Spigot";
  me=dqmStore_->get(s.c_str());  
  ChannSumm_DataIntegrityCheck_=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannSumm_DataIntegrityCheck_, debug_);
  if (!ChannSumm_DataIntegrityCheck_ & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;
  if (ChannSumm_DataIntegrityCheck_)
    ChannSumm_DataIntegrityCheck_->SetMinimum(0);

  char chararray[150];
  for (int i=0; i<NUMDCCS; i++) {
    sprintf(chararray,"Corruption/Channel Data Integrity/FED %03d Channel Integrity", i+700);
    s=subdir_+std::string(chararray);
    me=dqmStore_->get(s.c_str());  
    Chann_DataIntegrityCheck_[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Chann_DataIntegrityCheck_[i], debug_);
    if (!Chann_DataIntegrityCheck_[i] & (debug_>0)) std::cout <<"<HcalRawDataClient::analyze> "<<s<<" histogram does not exist!"<<std::endl;
    if (Chann_DataIntegrityCheck_[i])
      Chann_DataIntegrityCheck_[i]->SetMinimum(0);
  }
}
void HcalRawDataClient::fillProblemCountArray(void){
  if (debug_>1) std::cout <<"\t<HcalRawDataClient>::fillProblemCountArray(): getHardwareSpaceHistos()"<<std::endl;
  getHardwareSpaceHistos();
  float n=0.0;
  int dcc_=-999;

  bool CheckmeCDFErrorFound_                   = false; 
  bool CheckmeDCCEventFormatError_             = false;
  bool CheckmeOrNSynch_			       = false;
  bool CheckmeBCNSynch_			       = false;
  bool CheckmeEvtNumberSynch_		       = false;
  bool CheckLRBDataCorruptionIndicators_       = false;
  bool CheckHalfHTRDataCorruptionIndicators_   = false;
  bool CheckChannSumm_DataIntegrityCheck_      = false;
  bool CheckChann_DataIntegrityCheck_[NUMDCCS] = {false}; 

  if (meCDFErrorFound_!=0)                  CheckmeCDFErrorFound_                   = true;
  if (meDCCEventFormatError_!=0)            CheckmeDCCEventFormatError_             = true;
  if (meOrNSynch_!=0)                       CheckmeOrNSynch_			    = true;
  if (meBCNSynch_!=0)                       CheckmeBCNSynch_			    = true;
  if (meEvtNumberSynch_!=0)                 CheckmeEvtNumberSynch_		    = true;
  if (LRBDataCorruptionIndicators_!=0)      CheckLRBDataCorruptionIndicators_       = true;
  if (HalfHTRDataCorruptionIndicators_!=0)  CheckHalfHTRDataCorruptionIndicators_   = true;
  if (ChannSumm_DataIntegrityCheck_!=0)     CheckChannSumm_DataIntegrityCheck_      = true;

  int fed2offset=0;
  int fed3offset=0;
  int spg2offset=0;
  int spg3offset=0;
  int chn2offset=0;

  //Project all types of errors in these two plots onto
  //the x axis to get total errors per FED.
  TH1D* ProjXmeCDFErrorFound_       = 0;
  bool CheckProjXmeCDFErrorFound_ = false;
  if (CheckmeCDFErrorFound_)
    ProjXmeCDFErrorFound_=meCDFErrorFound_->ProjectionX();
  if (ProjXmeCDFErrorFound_!=0) CheckProjXmeCDFErrorFound_=true;
  TH1D* ProjXmeDCCEventFormatError_ = 0;
  bool CheckProjXmeDCCEventFormatError_ = false;
  if (CheckmeDCCEventFormatError_)
    ProjXmeDCCEventFormatError_=meDCCEventFormatError_->ProjectionX();
  if (ProjXmeDCCEventFormatError_!=0) CheckProjXmeDCCEventFormatError_ = true;

  for (int dccid=FEDNumbering::MINHCALFEDID; dccid<=FEDNumbering::MAXHCALFEDID; dccid++) {
    dcc_=dccid-FEDNumbering::MINHCALFEDID; // Numbering FEDS [0:31] is more useful for array indices.
    if (Chann_DataIntegrityCheck_[dcc_]!=0) 
      CheckChann_DataIntegrityCheck_[dcc_] = true;
    
    if (CheckProjXmeCDFErrorFound_) {
      n = ProjXmeCDFErrorFound_->GetBinContent(1+dcc_);
      if (n>0.0) mapDCCproblem(dcc_,n);
    }
    if (CheckProjXmeDCCEventFormatError_) {
      n = ProjXmeDCCEventFormatError_->GetBinContent(1+dcc_);
      if (n>0.0) mapDCCproblem(dcc_,n);
    }
  
    fed3offset = 1 + (4*dcc_); //3 bins, plus one of margin, each DCC (FED)
    fed2offset = 1 + (3*dcc_); //2 bins, plus one of margin, each DCC (FED)
    for (int spigot=0; spigot<NUMSPGS; spigot++) {
      
      if (CheckmeOrNSynch_) {
  	n = meOrNSynch_->GetBinContent(1+dcc_, 1+spigot);
  	if (n>0.0) mapHTRproblem(dcc_,spigot,n);
      }
      if (CheckmeBCNSynch_) {
  	n = meBCNSynch_->GetBinContent(1+dcc_, 1+spigot);
  	if (n>0.0) mapHTRproblem(dcc_,spigot,n);
      }
      if (CheckmeEvtNumberSynch_) {
  	n = meEvtNumberSynch_->GetBinContent(1+dcc_, 1+spigot);
  	if (n>0.0) mapHTRproblem(dcc_,spigot,n);
      }
      spg3offset = 1 + (4*spigot); //3 bins, plus one of margin, each spigot
      if (CheckLRBDataCorruptionIndicators_    ){
  	n=0.0; //Sum errors of all ten types 
  	n+=LRBDataCorruptionIndicators_->GetBinContent(fed3offset,
  						       spg3offset);
  	for (int xbin=1; xbin<=3; xbin++) {
  	  for (int ybin=1; ybin<=3; ybin++) {
  	    n+=LRBDataCorruptionIndicators_->GetBinContent(fed3offset+xbin,
  							   spg3offset+ybin);
  	  }
  	}
  	if (n>0.0) mapHTRproblem(dcc_,spigot,n);
      }
      if (CheckHalfHTRDataCorruptionIndicators_){
  	n=0.0; //Sum errors of all nine types 
  	for (int xbin=1; xbin<=3; xbin++) {
  	  for (int ybin=1; ybin<=3; ybin++) {
  	    n+=HalfHTRDataCorruptionIndicators_->GetBinContent(fed3offset+xbin,
  							       spg3offset+ybin);
  	  }
  	}
  	if (n>0.0) mapHTRproblem(dcc_,spigot,n);
      }
      spg2offset = 1 + (3*spigot); //2 bins, plus one of margin, each spigot
      if (CheckChann_DataIntegrityCheck_[dcc_] &&
  	  CheckChannSumm_DataIntegrityCheck_      ){
  	//Each spigot may be configured for its own number of TimeSlices, per event.
  	//Keep an array of the values:
  	numTS_[(dcc_*NUMSPGS)+spigot]=-1.0 * ChannSumm_DataIntegrityCheck_->GetBinContent(fed2offset,
											  spg2offset+1);
  	for (int chnnum=0; chnnum<HTRCHANMAX; chnnum++) {
  	  chn2offset = 1 + (3*chnnum); //2 bins, plus one of margin, each channel
  	  n = 0.0;
  	  //Sum errors of all types, 
  	  //but not !DV, at xbin==1, ybin==2.
  	  //Weight less if error can occur every timeslice
  	  // or between any two timeslices
  	  float tsFactor=numTS_[spigot +(dcc_*NUMSPGS)]; 
  	  float CRweight = 0.0;
  	  float Erweight = 0.0;
  	  if (tsFactor>0) {
  	    CRweight = (1.0 / (tsFactor-1.0));
  	    Erweight = (1.0 / (tsFactor    ));
  	  }
  	  int xbin=1; int ybin=1; // Timeslices per event check for error here
  	  n += Chann_DataIntegrityCheck_[dcc_]->GetBinContent(chn2offset+xbin,
  							      spg2offset+ybin);
  	  xbin=2; //move right one bin: CapID Rotation here
  	  n += CRweight * Chann_DataIntegrityCheck_[dcc_]->GetBinContent(chn2offset+xbin,
  									 spg2offset+ybin);
  	  ybin=2; //move up one bin: Er bit here
  	  n += Erweight * Chann_DataIntegrityCheck_[dcc_]->GetBinContent(chn2offset+xbin,
  									 spg2offset+ybin);
  	  if  (n>=0.0)
  	    mapChannproblem(dcc_,spigot,chnnum,n);
  	} //loop over channels
      } //check to see if FED had any channel problems  
    } //loop over spigot
  } //loop over dccid
}

void HcalRawDataClient::mapDCCproblem(int dcc, float n) {
  int myeta   = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up all affected cells.
  for (int i=hashup(dcc); 
       i<hashup(dcc)+(NUMSPGS*HTRCHANMAX); 
       i++) {
    HDI = hashedHcalDetId_[i];
    if (HDI==HcalDetId::Undefined) 
      continue;
    mydepth = HDI.depth();
    myphi   = HDI.iphi();
    myeta = CalcEtaBin(HDI.subdet(),
		       HDI.ieta(),
		       mydepth);
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4){
      if (problemcount[myeta][myphi-1][mydepth-1]< n)
	problemcount[myeta][myphi-1][mydepth-1]=n;

      //exclude the decommissioned HO ring2, except SiPMs 
      if(mydepth==4 && excludeHORing2_==true)
	if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemcount[myeta][myphi-1][mydepth-1] = 0.0;

      if (debug_>0)
	std::cout<<" mapDCCproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
    }
  }
}
void HcalRawDataClient::mapHTRproblem(int dcc, int spigot, float n) {
  int myeta = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up all affected cells.
  for (int i=hashup(dcc,spigot); 
       i<hashup(dcc,spigot)+(HTRCHANMAX); //nice, linear hash....
       i++) {
    HDI = hashedHcalDetId_[i];
    if (HDI==HcalDetId::Undefined) {
      continue;
    }
    mydepth = HDI.depth();
    myphi   = HDI.iphi();
    myeta = CalcEtaBin(HDI.subdet(),
		       HDI.ieta(),
		       mydepth);
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4){
      if (problemcount[myeta][myphi-1][mydepth-1]< n)
	problemcount[myeta][myphi-1][mydepth-1]=n;
      
      //exlcude the decommissioned HO ring2, except SiPMs 
      if(mydepth==4 && excludeHORing2_==true)
	if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemcount[myeta][myphi-1][mydepth-1] = 0.0;

      if (debug_>0)
	std::cout<<" mapHTRproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
    }    
  }
}   // void HcalRawDataClient::mapHTRproblem(...)

void HcalRawDataClient::mapChannproblem(int dcc, int spigot, int htrchan, float n) {
  int myeta = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up the affected cell.
  int i=hashup(dcc,spigot,htrchan); 
  HDI = HashToHDI(i);
  if (HDI==HcalDetId::Undefined) {
    return; // Do nothing at all, instead.
  } 
  mydepth = HDI.depth();
  myphi   = HDI.iphi();
  myeta = CalcEtaBin(HDI.subdet(),
		     HDI.ieta(),
		     mydepth);

  if (myeta>=0 && myeta<85 &&
      (myphi-1)>=0 && (myphi-1)<72 &&
      (mydepth-1)>=0 && (mydepth-1)<4){
    if (problemcount[myeta][myphi-1][mydepth-1]< n) {
      problemcount[myeta][myphi-1][mydepth-1]=n;

      //exlcude the decommissioned HO ring2, except SiPMs 
      if(mydepth==4 && excludeHORing2_==true)
	if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemcount[myeta][myphi-1][mydepth-1] = 0.0;

      if (debug_>0)
	std::cout<<" mapChannproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
    }
  }
}   // void HcalRawDataClient::mapChannproblem(...)


void HcalRawDataClient::normalizeHardwareSpaceHistos(void){
  /////Not ready for this yet.
//  // Get histograms that are used in testing
//  getHardwareSpaceHistos();
//
//  int fed2offset=0;
//  int spg2offset=0;
//  int chn2offset=0;
//  float tsFactor=1.0;
//  float val=0.0;
//
//  if (!ChannSumm_DataIntegrityCheck_) return;
//  //Normalize by the number of events each channel spake. (Handles ZS!)
//  for (int fednum=0;fednum<NUMDCCS;fednum++) {
//    fed2offset = 1 + (3*fednum); //2 bins, plus one of margin, each DCC 
//    for (int spgnum=0; spgnum<15; spgnum++) {
//      spg2offset = 1 + (3*spgnum); //2 bins, plus one of margin, each spigot
//      numTS_[(fednum*NUMSPGS)+spgnum]=ChannSumm_DataIntegrityCheck_->GetBinContent(fed2offset,
//										   spg2offset+1);
//
//      for (int xbin=1; xbin<=2; xbin++) {
//  	for (int ybin=1; ybin<=2; ybin++) {
//  	  val = ChannSumm_DataIntegrityCheck_->GetBinContent(fed2offset+xbin,
//  							     spg2offset+ybin);
//	  if ( (val) && (nevts_) ) {
//	    //Lower pair of bins don't scale with just the timesamples per event.
//	    if (ybin==2) tsFactor=numTS_[spgnum +(fednum*NUMSPGS)]; 
//	    else {
//	      if (xbin==2) tsFactor=numTS_[spgnum +(fednum*NUMSPGS)]-1;
//	      else tsFactor=1.0;
//	    }
//	    if (tsFactor)
//	      ChannSumm_DataIntegrityCheck_->SetBinContent(fed2offset+xbin,
//							   spg2offset+ybin,
//							   val/(nevts_*tsFactor));
//	    val=0.0;
//	  }
//  	}
//      }
//      //Clear the numTS, which clutter the final plot.
//      ChannSumm_DataIntegrityCheck_->SetBinContent(fed2offset  ,
//						   spg2offset  , 0.0);
//      ChannSumm_DataIntegrityCheck_->SetBinContent(fed2offset  ,
//						   spg2offset+1, 0.0);
//
//      if (!Chann_DataIntegrityCheck_[fednum]) continue;  
//      for (int chnnum=0; chnnum<24; chnnum++) {
//  	chn2offset = 1 + (3*chnnum); //2 bins, plus one of margin, each channel
//	if (! (Chann_DataIntegrityCheck_[fednum]))  
//	  continue;
//  	for (int xbin=1; xbin<=2; xbin++) {
//  	  for (int ybin=1; ybin<=2; ybin++) {
//  	    val = Chann_DataIntegrityCheck_[fednum]->GetBinContent(chn2offset+xbin,
//  								   spg2offset+ybin);
//  	    if ( (val) && (nevts_) ) {
//	      //Lower pair of bins don't scale with just the timesamples per event.
//	      if (ybin==2) tsFactor=numTS_[spgnum +(fednum*NUMSPGS)]; 
//	      else {
//		if (xbin==2) tsFactor=numTS_[spgnum +(fednum*NUMSPGS)]-1;
//		else tsFactor=1.0;
//	      }
//	      if (tsFactor)
//		Chann_DataIntegrityCheck_[fednum]->SetBinContent(chn2offset+xbin,
//								 spg2offset+ybin,
//								 val/(nevts_*tsFactor));
//	    }
//  	  }
//  	}
//	//Remove the channel's event count from sight.
//	Chann_DataIntegrityCheck_[fednum]->SetBinContent(chn2offset,
//							 spg2offset,0.0);
//      }
//    }
//  }  
}

HcalRawDataClient::~HcalRawDataClient()
{}
