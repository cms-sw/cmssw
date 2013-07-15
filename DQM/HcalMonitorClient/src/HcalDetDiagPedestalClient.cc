#include "DQM/HcalMonitorClient/interface/HcalDetDiagPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include <iostream>

/*
 * \file HcalDetDiagPedestalClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.12 $
 * \author J. Temple
 * \brief Hcal DetDiagPedestal Client class
 */


HcalDetDiagPedestalClient::HcalDetDiagPedestalClient(std::string myname)
{
  name_=myname;   status=0;
  needLogicalMap_=true;
}

HcalDetDiagPedestalClient::HcalDetDiagPedestalClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("DetDiagPedestalFolder","DetDiagPedestalMonitor_Hcal/"); // DetDiagPedestalMonitor_Hcal/
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("DetDiagPedestal_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("DetDiagPedestal_BadChannelStatusMask",
			   ps.getUntrackedParameter<int>("BadChannelStatusMask",(1<<HcalChannelStatus::HcalCellDead)));
  
  minerrorrate_ = ps.getUntrackedParameter<double>("DetDiagPedestal_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.05));
  minevents_    = ps.getUntrackedParameter<int>("DetDiagPedestal_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
  needLogicalMap_=true;
}

void HcalDetDiagPedestalClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalDetDiagPedestalClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalDetDiagPedestalClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalDetDiagPedestalClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  double totalevents=0;
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;

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
  enoughevents_=true;
  // Get histograms that are used in testing
  // currently none used,

  std::vector<std::string> name = HcalEtaPhiHistNames();

  // This is a sample of how to get a histogram from the task that can then be used for evaluation purposes
  TH2F* PedestalsMissing[4];
  TH2F* PedestalsUnstable[4];
  TH2F* PedestalsBadMean[4];
  TH2F* PedestalsBadRMS[4];
  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      // Assume that histograms aren't found
      PedestalsMissing[i]=0;
      PedestalsUnstable[i]=0;
      PedestalsBadMean[i]=0;
      PedestalsBadRMS[i]=0;
      std::string s=subdir_+name[i]+" Problem Missing Channels";
      me=dqmStore_->get(s.c_str());
      if (me!=0) PedestalsMissing[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, PedestalsMissing[i], debug_);
      else 
	{
	  if (debug_>0) 
	    std::cout <<"<HcalDetDiagPedestalClient::calcluateProblems> could not get histogram '"<<s<<"'"<<std::endl;
	}

      s=subdir_+name[i]+" Problem Unstable Channels";
      me=dqmStore_->get(s.c_str());
      if (me!=0) PedestalsUnstable[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, PedestalsUnstable[i], debug_);
      else if (debug_>0) std::cout <<"<HcalDetDiagPedestalClient::calculateProblems> could not get histogram '"<<s<<"'"<<std::endl;
      s=subdir_+name[i]+" Problem Bad Pedestal Value";
      me=dqmStore_->get(s.c_str());
      if (me!=0) PedestalsBadMean[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, PedestalsBadMean[i], debug_);
      else if (debug_>0) std::cout <<"<HcalDetDiagPedestalClient::calculateProblems> could not get histogram '"<<s<<"'"<<std::endl;
      s=subdir_+name[i]+" Problem Bad Rms Value";
      me=dqmStore_->get(s.c_str());
      if (me!=0) PedestalsBadRMS[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, PedestalsBadRMS[i], debug_);
      else if (debug_>0) std::cout <<"<HcalDetDiagPedestalClient::calculateProblems> could not get histogram '"<<s<<"'"<<std::endl;
    }      

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;
    
      //totalevents=DigiPresentByDepth[d]->GetBinContent(0);
      totalevents=0;
      // Check underflow bins for events processed
      if (PedestalsMissing[d]!=0) totalevents = PedestalsMissing[d]->GetBinContent(0);
      else if (PedestalsUnstable[d]!=0) totalevents = PedestalsUnstable[d]->GetBinContent(0);
      else if (PedestalsBadMean[d]!=0) totalevents = PedestalsBadMean[d]->GetBinContent(0);
      else if (PedestalsBadRMS[d]!=0) totalevents = PedestalsBadRMS[d]->GetBinContent(0);
      //if (totalevents==0 || totalevents<minevents_) continue;
      
      totalevents=1; // temporary value pending removal of normalization from task
      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;
	      if (PedestalsMissing[d]!=0) problemvalue += PedestalsMissing[d]->GetBinContent(eta+1,phi+1)*1./totalevents;
	      if (PedestalsUnstable[d]!=0) problemvalue += PedestalsUnstable[d]->GetBinContent(eta+1,phi+1)*1./totalevents;
	      if (PedestalsBadMean[d]!=0) problemvalue += PedestalsBadMean[d]->GetBinContent(eta+1,phi+1)*1./totalevents;
	      if (PedestalsBadRMS[d]!=0) problemvalue += PedestalsBadRMS[d]->GetBinContent(eta+1,phi+1)*1./totalevents;

	      if (problemvalue==0) continue;
	      // problem value is a rate; we can normalize it here
	      problemvalue = std::min(1.,problemvalue);
	      
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
      if (debug_>0) std::cout <<"<HcalDetDiagPedestalClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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

void HcalDetDiagPedestalClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalDetDiagPedestalClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalDetDiagPedestalClient::endJob(){}

void HcalDetDiagPedestalClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalDetDiagPedestalClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemDetDiagPedestal",
				 " Problem DetDiagPedestal Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_DetDiagPedestal");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem DetDiagPedestal Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;
}

void HcalDetDiagPedestalClient::endRun(void){analyze();}

void HcalDetDiagPedestalClient::setup(void){}
void HcalDetDiagPedestalClient::cleanup(void){}

bool HcalDetDiagPedestalClient::hasErrors_Temp(void)
{
    if(status&2) return true;
    return false;

  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalDetDiagPedestalClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalDetDiagPedestalClient::hasWarnings_Temp(void){
   if(status&1) return true;
   return false;
}
bool HcalDetDiagPedestalClient::hasOther_Temp(void){return false;}
bool HcalDetDiagPedestalClient::test_enabled(void){return true;}


void HcalDetDiagPedestalClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

} //void HcalDetDiagPedestalClient::updateChannelStatus

static void printTableHeader(ofstream& file,std::string  header){
  file << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">"<< std::endl;
     file << "<head>"<< std::endl;
     file << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>"<< std::endl;
     file << "<title>"<< header <<"</title>"<< std::endl;
     file << "<style type=\"text/css\">"<< std::endl;
     file << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"<< std::endl;
     file << "   td.s0 { font-family: arial, arial ce, helvetica; }"<< std::endl;
     file << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< std::endl;
     file << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }"<< std::endl;
     file << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }"<< std::endl;
     file << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }"<< std::endl;
     file << "</style>"<< std::endl;
     file << "<body>"<< std::endl;
     file << "<table>"<< std::endl;
}

static void printTableLine(ofstream& file,int ind,HcalDetId& detid,HcalFrontEndId& lmap_entry,HcalElectronicsId &emap_entry,std::string comment=""){
   if(ind==0){
     file << "<tr>";
     file << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     file << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     file << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     file << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     file << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     file << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     file << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     if(comment[0]!=0) file << "<td class=\"s1\" align=\"center\">Comment</td>"   << std::endl;
     file << "</tr>"   << std::endl;
   }
   std::string raw_class;
   file << "<tr>"<< std::endl;
   if((ind%2)==1){
      raw_class="<td class=\"s2\" align=\"center\">";
   }else{
      raw_class="<td class=\"s3\" align=\"center\">";
   }
   file << "<td class=\"s4\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
   file << raw_class<< detid.ieta()<<"</td>"<< std::endl;
   file << raw_class<< detid.iphi()<<"</td>"<< std::endl;
   file << raw_class<< detid.depth() <<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.rbx()<<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.rm() <<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.pixel()<<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.rmFiber() <<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.fiberChannel()<<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.qieCard() <<"</td>"<< std::endl;
   file << raw_class<< lmap_entry.adc()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.readoutVMECrateId()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.dccid()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.spigot()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.fiberIndex()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.htrSlot()<<"</td>"<< std::endl;
   file << raw_class<< emap_entry.htrTopBottom()<<"</td>"<< std::endl;
   if(comment[0]!=0) file << raw_class<< comment<<"</td>"<< std::endl;
}
static void printTableTail(ofstream& file){
     file << "</table>"<< std::endl;
     file << "</body>"<< std::endl;
     file << "</html>"<< std::endl;
}

bool HcalDetDiagPedestalClient::validHtmlOutput(){
  std::string s=subdir_+"HcalDetDiagPedestalMonitor Event Number";
  MonitorElement *me = dqmStore_->get(s.c_str());
  int n=0;
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &n);
  }
  if(n<100) return false;
  return true;
}

void HcalDetDiagPedestalClient::htmlOutput(std::string htmlDir){
int  MissingCnt=0,UnstableCnt=0,BadCnt=0; 
int  HBP[4]={0,0,0,0},HBM[4]={0,0,0,0},HEP[4]={0,0,0,0},HEM[4]={0,0,0,0},HFP[4]={0,0,0,0},HFM[4]={0,0,0,0},HO[4] ={0,0,0,0}; 
int  newHBP[4]={0,0,0,0},newHBM[4]={0,0,0,0},newHEP[4]={0,0,0,0},newHEM[4]={0,0,0,0};
int  newHFP[4]={0,0,0,0},newHFM[4]={0,0,0,0},newHO[4] ={0,0,0,0}; 
 if (debug_>0) std::cout << "<HcalDetDiagPedestalClient::htmlOutput> Preparing  html output ..." << std::endl;
  if(!dqmStore_) return;

  HcalElectronicsMap emap=logicalMap_->generateHcalElectronicsMap();
  TH2F *Missing_val[4],*Unstable_val[4],*BadPed_val[4],*BadRMS_val[4];
  MonitorElement* me;


  TH1F *PedestalsAve4HB=0;
  TH1F *PedestalsAve4HE=0;
  TH1F *PedestalsAve4HO=0;
  TH1F *PedestalsAve4HF=0;
  TH1F *PedestalsAve4Simp=0;
 
  TH1F *PedestalsAve4HBref=0;
  TH1F *PedestalsAve4HEref=0;
  TH1F *PedestalsAve4HOref=0;
  TH1F *PedestalsAve4HFref=0;
  TH1F *PedestalsRmsHB=0;
  TH1F *PedestalsRmsHE=0;
  TH1F *PedestalsRmsHO=0;
  TH1F *PedestalsRmsHF=0;
  TH1F *PedestalsRmsSimp=0;
  
  TH1F *PedestalsRmsHBref=0;
  TH1F *PedestalsRmsHEref=0;
  TH1F *PedestalsRmsHOref=0;
  TH1F *PedestalsRmsHFref=0;
  
  TH2F *Pedestals2DRmsHBHEHF=0;
  TH2F *Pedestals2DRmsHO=0;
  TH2F *Pedestals2DHBHEHF=0;
  TH2F *Pedestals2DHO=0;
  TH2F *Pedestals2DErrorHBHEHF=0;
  TH2F *Pedestals2DErrorHO=0;

  std::string s=subdir_+"Summary Plots/HB Pedestal Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str());
  if(me!=0) PedestalsAve4HB=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HB, debug_); else return;
  s=subdir_+"Summary Plots/HE Pedestal Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HE=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HE, debug_);  else return; 
  s=subdir_+"Summary Plots/HO Pedestal Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HO=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HO, debug_);  else return; 
  s=subdir_+"Summary Plots/HF Pedestal Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HF, debug_);  else return; 
  s=subdir_+"Summary Plots/SIPM Pedestal Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4Simp=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4Simp, debug_); else return; 
 
  s=subdir_+"Summary Plots/HB Pedestal-Reference Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HBref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HBref, debug_); else return;  
  s=subdir_+"Summary Plots/HE Pedestal-Reference Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HEref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HEref, debug_); else return; 
  s=subdir_+"Summary Plots/HO Pedestal-Reference Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HOref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HOref, debug_); else return;  
  s=subdir_+"Summary Plots/HF Pedestal-Reference Distribution (average over 4 caps)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsAve4HFref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsAve4HFref, debug_); else return;  
   
  s=subdir_+"Summary Plots/HB Pedestal RMS Distribution (individual cap)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHB=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHB, debug_);  else return; 
  s=subdir_+"Summary Plots/HE Pedestal RMS Distribution (individual cap)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHE=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHE, debug_);  else return; 
  s=subdir_+"Summary Plots/HO Pedestal RMS Distribution (individual cap)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHO=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHO, debug_);  else return; 
  s=subdir_+"Summary Plots/HF Pedestal RMS Distribution (individual cap)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHF, debug_);  else return; 
  s=subdir_+"Summary Plots/SIPM Pedestal RMS Distribution (individual cap)"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsSimp=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsSimp, debug_);  else return; 
   
  s=subdir_+"Summary Plots/HB Pedestal_rms-Reference_rms Distribution"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHBref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHBref, debug_);  else return; 
  s=subdir_+"Summary Plots/HE Pedestal_rms-Reference_rms Distribution"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHEref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHEref, debug_);  else return; 
  s=subdir_+"Summary Plots/HO Pedestal_rms-Reference_rms Distribution"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHOref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHOref, debug_);  else return; 
  s=subdir_+"Summary Plots/HF Pedestal_rms-Reference_rms Distribution"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) PedestalsRmsHFref=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, PedestalsRmsHFref, debug_);  else return; 
     
  s=subdir_+"Summary Plots/HBHEHF pedestal mean map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DHBHEHF=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DHBHEHF, debug_); else return;  
  s=subdir_+"Summary Plots/HO pedestal mean map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DHO=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DHO, debug_);  else return; 
  s=subdir_+"Summary Plots/HBHEHF pedestal rms map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DRmsHBHEHF=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DRmsHBHEHF, debug_); else return;  
  s=subdir_+"Summary Plots/HO pedestal rms map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DRmsHO=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DRmsHO, debug_);  else return; 
  s=subdir_+"Summary Plots/HBHEHF pedestal problems map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DErrorHBHEHF=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DErrorHBHEHF, debug_);  else return; 
  s=subdir_+"Summary Plots/HO pedestal problems map"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) Pedestals2DErrorHO=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Pedestals2DErrorHO, debug_); else return;  


  std::vector<std::string> name = HcalEtaPhiHistNames();
  for(int i=0;i<4;++i){
      Missing_val[i]=Unstable_val[i]=BadPed_val[i]=BadRMS_val[i]=0;
      std::string s=subdir_+"Plots for client/"+name[i]+" Missing channels";
      me=dqmStore_->get(s.c_str());
      if (me!=0) Missing_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Missing_val[i], debug_); else return;  
      s=subdir_+"Plots for client/"+name[i]+" Channel instability value";
      me=dqmStore_->get(s.c_str());
      if (me!=0) Unstable_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Unstable_val[i], debug_); else return;  
      s=subdir_+"Plots for client/"+name[i]+" Bad Pedestal-Ref Value";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadPed_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadPed_val[i], debug_); else return;  
      s=subdir_+"Plots for client/"+name[i]+" Bad Rms-ref Value";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadRMS_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadRMS_val[i], debug_); else return;  
  }
  // Calculate problems 
  for(int d=0;d<4;++d){
      int etabins=Missing_val[d]->GetNbinsX();
      int phibins=Missing_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          HcalSubdetector subdet=HcalEmpty;
          if(isHB(eta,d+1))subdet=HcalBarrel;
	     else if (isHE(eta,d+1)) subdet=HcalEndcap;
	     else if (isHF(eta,d+1)) subdet=HcalForward;
	     else if (isHO(eta,d+1)) subdet=HcalOuter;
	  HcalDetId hcalid(subdet, ieta, phi+1, (int)(d+1));
          float val=Missing_val[d]->GetBinContent(eta+1,phi+1);
	  if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[0]++;}else{ HBM[0]++;} MissingCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[0]++;}else{ newHBM[0]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[0]++;}else{ HEM[0]++;} MissingCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[0]++;}else{ newHEM[0]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[0]++;}else{ HFM[0]++;} MissingCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[0]++;}else{ newHFM[0]++;}}
            }	
            if(subdet==HcalOuter){
               HO[0]++;MissingCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[0]++;}
            }	
         }
         val=Unstable_val[d]->GetBinContent(eta+1,phi+1);
	 if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[1]++;}else{ HBM[1]++;} UnstableCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[1]++;}else{ newHBM[1]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[1]++;}else{ HEM[1]++;} UnstableCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[1]++;}else{ newHEM[1]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[1]++;}else{ HFM[1]++;} UnstableCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[1]++;}else{ newHFM[1]++;}}
            }	
            if(subdet==HcalOuter){
               HO[1]++;UnstableCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[1]++;}
            }	
         }
         val=BadPed_val[d]->GetBinContent(eta+1,phi+1);
	 if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[2]++;}else{ HBM[2]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[2]++;}else{ newHBM[2]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[2]++;}else{ HEM[2]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[2]++;}else{ newHEM[2]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[2]++;}else{ HFM[2]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[2]++;}else{ newHFM[2]++;}}
            }	
            if(subdet==HcalOuter){
               HO[2]++;BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[2]++;}
            }	
         }
         val=BadRMS_val[d]->GetBinContent(eta+1,phi+1);
	 if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[3]++;}else{ HBM[3]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[3]++;}else{ newHBM[3]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[3]++;}else{ HEM[3]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[3]++;}else{ newHEM[3]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[3]++;}else{ HFM[3]++;} BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[3]++;}else{ newHFM[3]++;}}
            }	
            if(subdet==HcalOuter){
               HO[3]++;BadCnt++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[3]++;}
            }	
         }
      } 
  } 


  ofstream badMissing; 
  badMissing.open((htmlDir+"bad_missing_table.html").c_str());
  printTableHeader(badMissing,"Missing Channels list");
  ofstream badUnstable; 
  badUnstable.open((htmlDir+"bad_unstable_table.html").c_str());
  printTableHeader(badUnstable,"Unstable Channels list");
  ofstream badPedRMS; 
  badPedRMS.open((htmlDir+"bad_badpedrms_table.html").c_str());
  printTableHeader(badPedRMS,"Missing Channels list");

  int cnt=0;
  if((HBP[0]+HBP[0])>0 && (HBM[0]+HBP[0])!=(1296*2)){
    badMissing << "<tr><td align=\"center\"><h3>"<< "HB" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Missing_val[d]->GetNbinsX();
      int phibins=Missing_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHB(eta,d+1)) continue;
          float val=Missing_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalBarrel,ieta,phi+1,d+1);
          std::string s=" ";
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s="Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badMissing,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HEP[0]+HEP[0])>0 && (HEM[0]+HEP[0])!=(1296*2)){
    badMissing << "<tr><td align=\"center\"><h3>"<< "HE" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Missing_val[d]->GetNbinsX();
      int phibins=Missing_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHE(eta,d+1)) continue;
          float val=Missing_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalEndcap,ieta,phi+1,d+1);
          std::string s=" ";
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s="Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badMissing,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if(HO[0]>0 && HO[0]!=2160){
    badMissing << "<tr><td align=\"center\"><h3>"<< "HO" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Missing_val[d]->GetNbinsX();
      int phibins=Missing_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHO(eta,d+1)) continue;
          float val=Missing_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalOuter,ieta,phi+1,d+1);
          std::string s=" ";
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s="Known problem";}
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badMissing,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HFP[0]+HFP[0])>0 && (HFM[0]+HFP[0])!=(864*2)){
    badMissing << "<tr><td align=\"center\"><h3>"<< "HF" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Missing_val[d]->GetNbinsX();
      int phibins=Missing_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHF(eta,d+1)) continue;
          float val=Missing_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalForward,ieta,phi+1,d+1);
          std::string s=" ";
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s="Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badMissing,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
/////////////////////////////////////////////////////////////////////////////////////
  cnt=0;
  if((HBP[1]+HBP[1])>0){
    badUnstable << "<tr><td align=\"center\"><h3>"<< "HB" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Unstable_val[d]->GetNbinsX();
      int phibins=Unstable_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHB(eta,d+1)) continue;
          float val=Unstable_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalBarrel,ieta,phi+1,d+1);
          char comment[100]; sprintf(comment,"Missing in %.3f%% of events\n",(1.0-val)*100.0);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badUnstable,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HEP[1]+HEP[1])>0){
    badUnstable << "<tr><td align=\"center\"><h3>"<< "HE" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Unstable_val[d]->GetNbinsX();
      int phibins=Unstable_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHE(eta,d+1)) continue;
          float val=Unstable_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalEndcap,ieta,phi+1,d+1);
          char comment[100]; sprintf(comment,"Missing in %.3f%% of events\n",(1.0-val)*100.0);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badUnstable,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if(HO[1]>0){
    badUnstable << "<tr><td align=\"center\"><h3>"<< "HO" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Unstable_val[d]->GetNbinsX();
      int phibins=Unstable_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHO(eta,d+1)) continue;
          float val=Unstable_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalOuter,ieta,phi+1,d+1);
          char comment[100]; sprintf(comment,"Missing in %.3f%% of events\n",(1.0-val)*100.0);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badUnstable,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HFP[1]+HFP[1])>0){
    badUnstable << "<tr><td align=\"center\"><h3>"<< "HF" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=Unstable_val[d]->GetNbinsX();
      int phibins=Unstable_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHF(eta,d+1)) continue;
          float val=Unstable_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalForward,ieta,phi+1,d+1);
          char comment[100]; sprintf(comment,"Missing in %.3f%% of events\n",(1.0-val)*100.0);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badUnstable,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
/////////////////////////////////////////////////////////////////////////////////////
  cnt=0;
  if((HBP[2]+HBP[2]+HBP[3]+HBP[3])>0){
    badPedRMS << "<tr><td align=\"center\"><h3>"<< "HB" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=BadPed_val[d]->GetNbinsX();
      int phibins=BadPed_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHB(eta,d+1)) continue;
          float val1=BadPed_val[d]->GetBinContent(eta+1,phi+1);
          float val2=BadRMS_val[d]->GetBinContent(eta+1,phi+1);
	  if(val1==0 && val2==0) continue;
          HcalDetId hcalid(HcalBarrel,ieta,phi+1,d+1);
	  char comment[100]; 
	  if(val1!=0) sprintf(comment,"Ped-Ref=%.2f",val1);
	  if(val2!=0) sprintf(comment,"Rms-Ref=%.2f",val2);
	  if(val1!=0 && val2!=0) sprintf(comment,"Ped-Ref=%.2f,Rms-Ref=%.2f",val1,val2);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badPedRMS,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HEP[2]+HEP[2]+HEP[3]+HEP[3])>0){
    badPedRMS << "<tr><td align=\"center\"><h3>"<< "HE" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=BadPed_val[d]->GetNbinsX();
      int phibins=BadPed_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHE(eta,d+1)) continue;
          float val1=BadPed_val[d]->GetBinContent(eta+1,phi+1);
          float val2=BadRMS_val[d]->GetBinContent(eta+1,phi+1);
	  if(val1==0 && val2==0) continue;
          HcalDetId hcalid(HcalEndcap,ieta,phi+1,d+1);
	  char comment[100]; 
	  if(val1!=0) sprintf(comment,"Ped-Ref=%.2f",val1);
	  if(val2!=0) sprintf(comment,"Rms-Ref=%.2f",val2);
	  if(val1!=0 && val2!=0) sprintf(comment,"Ped-Ref=%.2f,Rms-Ref=%.2f",val1,val2);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badPedRMS,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HO[2]+HO[3])>0){
    badPedRMS << "<tr><td align=\"center\"><h3>"<< "HO" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=BadPed_val[d]->GetNbinsX();
      int phibins=BadPed_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHO(eta,d+1)) continue;
          float val1=BadPed_val[d]->GetBinContent(eta+1,phi+1);
          float val2=BadRMS_val[d]->GetBinContent(eta+1,phi+1);
	  if(val1==0 && val2==0) continue;
          HcalDetId hcalid(HcalOuter,ieta,phi+1,d+1);
	  char comment[100]; 
	  if(val1!=0) sprintf(comment,"Ped-Ref=%.2f",val1);
	  if(val2!=0) sprintf(comment,"Rms-Ref=%.2f",val2);
	  if(val1!=0 && val2!=0) sprintf(comment,"Ped-Ref=%.2f,Rms-Ref=%.2f",val1,val2);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badPedRMS,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }
  cnt=0;
  if((HFP[2]+HFP[2]+HFP[3]+HFP[3])>0){
    badPedRMS << "<tr><td align=\"center\"><h3>"<< "HF" <<"</h3></td></tr>" << std::endl;
    for(int d=0;d<4;++d){
      int etabins=BadPed_val[d]->GetNbinsX();
      int phibins=BadPed_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHF(eta,d+1)) continue;
          float val1=BadPed_val[d]->GetBinContent(eta+1,phi+1);
          float val2=BadRMS_val[d]->GetBinContent(eta+1,phi+1);
	  if(val1==0 && val2==0) continue;
          HcalDetId hcalid(HcalForward,ieta,phi+1,d+1);
	  char comment[100]; 
	  if(val1!=0) sprintf(comment,"Ped-Ref=%.2f",val1);
	  if(val2!=0) sprintf(comment,"Rms-Ref=%.2f",val2);
	  if(val1!=0 && val2!=0) sprintf(comment,"Ped-Ref=%.2f,Rms-Ref=%.2f",val1,val2);
          std::string s=comment;
          if(badstatusmap.find(hcalid)!=badstatusmap.end()){ s+=",Known problem";}	
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
	  printTableLine(badPedRMS,cnt++,hcalid,lmap_entry,emap_entry,s);
      } 
    } 
  }

  printTableTail(badMissing);
  badMissing.close();
  printTableTail(badUnstable);
  badUnstable.close();
  printTableTail(badPedRMS);
  badPedRMS.close();

  int ievt_ = -1,runNo=-1;
  std::string ref_run;
  s=subdir_+"HcalDetDiagPedestalMonitor Event Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  s=subdir_+"HcalDetDiagPedestalMonitor Run Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &runNo);
  } 
  s=subdir_+"HcalDetDiagLaserMonitor Reference Run";
  me = dqmStore_->get(s.c_str());
  if(me) {
    std::string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
  }

  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
 
  TCanvas *can=new TCanvas("HcalDetDiagPedestalClient","HcalDetDiagPedestalClient",0,0,500,350);
  can->cd();

  ofstream htmlFile;
  std::string outfile=htmlDir+name_+".html";
  htmlFile.open(outfile.c_str());
  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Detector Diagnostics Pedestal Monitor</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<style type=\"text/css\">"<< std::endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FF7700; text-align: center;}"<< std::endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< std::endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: red; }"<< std::endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: yellow; }"<< std::endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: green; }"<< std::endl;
  htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: silver; }"<< std::endl;
  std::string state[4]={"<td class=\"s2\" align=\"center\">",
			"<td class=\"s3\" align=\"center\">",
			"<td class=\"s4\" align=\"center\">",
			"<td class=\"s5\" align=\"center\">"};
  htmlFile << "</style>"<< std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics Pedestal Monitor</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  /////////////////////////////////////////// 
  htmlFile << "<table width=100% border=1>" << std::endl;
  htmlFile << "<tr>" << std::endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">SebDet</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Missing</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Unstable</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad |Ped-Ref|</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad |Rms-Ref|</td>" << std::endl;
  htmlFile << "</tr><tr>" << std::endl;
  int ind1=0,ind2=0,ind3=0,ind4=0;
  htmlFile << "<td class=\"s1\" align=\"center\">HB+</td>" << std::endl;
  ind1=3; if(newHBP[0]==0) ind1=2; if(newHBP[0]>0 && newHBP[0]<=12) ind1=1; if(newHBP[0]>=12 && newHBP[0]<1296) ind1=0; 
  ind2=3; if(newHBP[1]==0) ind2=2; if(newHBP[1]>0)  ind2=1; if(newHBP[1]>21)  ind2=0; 
  ind3=3; if(newHBP[2]==0) ind3=2; if(newHBP[2]>0)  ind3=1; if(newHBP[2]>21)  ind3=0;
  ind4=3; if(newHBP[3]==0) ind4=2; if(newHBP[3]>0)  ind4=1; if(newHBP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;  
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HBP[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HBP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HBP[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HBP[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HB-</td>" << std::endl;
  ind1=3; if(newHBM[0]==0) ind1=2; if(newHBM[0]>0 && newHBM[0]<=12) ind1=1; if(newHBM[0]>=12 && newHBM[0]<1296) ind1=0; 
  ind2=3; if(newHBM[1]==0) ind2=2; if(newHBM[1]>0)  ind2=1; if(newHBM[1]>21)  ind2=0; 
  ind3=3; if(newHBM[2]==0) ind3=2; if(newHBM[2]>0)  ind3=1; if(newHBM[2]>21)  ind3=0;
  ind4=3; if(newHBM[3]==0) ind4=2; if(newHBM[3]>0)  ind4=1; if(newHBM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HBM[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HBM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HBM[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HBM[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE+</td>" << std::endl;
  ind1=3; if(newHEP[0]==0) ind1=2; if(newHEP[0]>0 && newHEP[0]<=12) ind1=1; if(newHEP[0]>=12 && newHEP[0]<1296) ind1=0; 
  ind2=3; if(newHEP[1]==0) ind2=2; if(newHEP[1]>0)  ind2=1; if(newHEP[1]>21)  ind2=0; 
  ind3=3; if(newHEP[2]==0) ind3=2; if(newHEP[2]>0)  ind3=1; if(newHEP[2]>21)  ind3=0;
  ind4=3; if(newHEP[3]==0) ind4=2; if(newHEP[3]>0)  ind4=1; if(newHEP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HEP[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HEP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HEP[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HEP[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE-</td>" << std::endl;
  ind1=3; if(newHEM[0]==0) ind1=2; if(newHEM[0]>0 && newHEM[0]<=12) ind1=1; if(newHEM[0]>=12 && newHEM[0]<1296) ind1=0; 
  ind2=3; if(newHEM[1]==0) ind2=2; if(newHEM[1]>0)  ind2=1; if(newHEM[1]>21)  ind2=0; 
  ind3=3; if(newHEM[2]==0) ind3=2; if(newHEM[2]>0)  ind3=1; if(newHEM[2]>21)  ind3=0;
  ind4=3; if(newHEM[3]==0) ind4=2; if(newHEM[3]>0)  ind4=1; if(newHEM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HEM[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HEM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HEM[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HEM[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF+</td>" << std::endl;
  ind1=3; if(newHFP[0]==0) ind1=2; if(newHFP[0]>0 && newHFP[0]<=12) ind1=1; if(newHFP[0]>=12 && newHFP[0]<864) ind1=0; 
  ind2=3; if(newHFP[1]==0) ind2=2; if(newHFP[1]>0)  ind2=1; if(newHFP[1]>21)  ind2=0; 
  ind3=3; if(newHFP[2]==0) ind3=2; if(newHFP[2]>0)  ind3=1; if(newHFP[2]>21)  ind3=0;
  ind4=3; if(newHFP[3]==0) ind4=2; if(newHFP[3]>0)  ind4=1; if(newHFP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HFP[0] <<" (864)</td>" << std::endl;
  htmlFile << state[ind2] << HFP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HFP[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HFP[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF-</td>" << std::endl;
  ind1=3; if(newHFM[0]==0) ind1=2; if(newHFM[0]>0 && newHFM[0]<=12) ind1=1; if(newHFM[0]>=12 && newHFM[0]<864) ind1=0; 
  ind2=3; if(newHFM[1]==0) ind2=2; if(newHFM[1]>0)  ind2=1; if(newHFM[1]>21)  ind2=0; 
  ind3=3; if(newHFM[2]==0) ind3=2; if(newHFM[2]>0)  ind3=1; if(newHFM[2]>21)  ind3=0;
  ind4=3; if(newHFM[3]==0) ind4=2; if(newHFM[3]>0)  ind4=1; if(newHFM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HFM[0] <<" (864)</td>" << std::endl;
  htmlFile << state[ind2] << HFM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HFM[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HFM[3] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HO</td>" << std::endl;
  ind1=3; if(newHO[0]==0) ind1=2; if(newHO[0]>0 && newHO[0]<=12) ind1=1; if(newHO[0]>=12 && newHO[0]<2160) ind1=0; 
  ind2=3; if(newHO[1]==0) ind2=2; if(newHO[1]>0)  ind2=1; if(newHO[1]>21)  ind2=0; 
  ind3=3; if(newHO[2]==0) ind3=2; if(newHO[2]>0)  ind3=1; if(newHO[2]>21)  ind3=0;
  ind4=3; if(newHO[3]==0) ind4=2; if(newHO[3]>0)  ind4=1; if(newHO[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HO[0] <<" (2160)</td>" << std::endl;
  htmlFile << state[ind2] << HO[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HO[2] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HO[3] <<"</td>" << std::endl;

  htmlFile << "</tr></table>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  /////////////////////////////////////////// 
  if((MissingCnt+UnstableCnt+BadCnt)>0){
      htmlFile << "<table width=100% border=1><tr>" << std::endl;
      if(MissingCnt>0)  htmlFile << "<td><a href=\"" << "bad_missing_table.html" <<"\">list of missing channels</a></td>";
      if(UnstableCnt>0) htmlFile << "<td><a href=\"" << "bad_unstable_table.html" <<"\">list of unstable channels</a></td>";
      if(BadCnt>0)      htmlFile << "<td><a href=\"" << "bad_badpedrms_table.html" <<"\">list of bad pedestal/rms channels</a></td>";
      htmlFile << "</tr></table>" << std::endl;
  }
  can->SetGridy();
  can->SetGridx();
  can->SetLogy(0); 



  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">Summary plots</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  Pedestals2DHBHEHF->SetStats(0);
  Pedestals2DHBHEHF->SetMaximum(5);
  Pedestals2DHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_pedestal_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_pedestal_map.gif\" alt=\"hbhehf pedestal mean map\">   </td>" << std::endl;
  Pedestals2DHO->SetStats(0);
  Pedestals2DHO->SetMaximum(5);
  Pedestals2DHO->SetNdivisions(36,"Y");
  Pedestals2DHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_pedestal_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_map.gif\" alt=\"ho pedestal mean map\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  Pedestals2DRmsHBHEHF->SetStats(0);
  Pedestals2DRmsHBHEHF->SetMaximum(2);
  Pedestals2DRmsHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DRmsHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_rms_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_rms_map.gif\" alt=\"hbhehf pedestal rms map\">   </td>" << std::endl;
  Pedestals2DRmsHO->SetStats(0);
  Pedestals2DRmsHO->SetMaximum(2);
  Pedestals2DRmsHO->SetNdivisions(36,"Y");
  Pedestals2DRmsHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_rms_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_rms_map.gif\" alt=\"ho pedestal rms map\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  Pedestals2DErrorHBHEHF->SetStats(0);
  Pedestals2DErrorHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DErrorHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_error_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_error_map.gif\" alt=\"hbhehf pedestal error map\">   </td>" << std::endl;
  Pedestals2DErrorHO->SetStats(0);
  Pedestals2DErrorHO->SetNdivisions(36,"Y");
  Pedestals2DErrorHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_error_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_error_map.gif\" alt=\"ho pedestal error map\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  
  ///////////////////////////////////////////   
  htmlFile << "<h2 align=\"center\">HB Pedestal plots (Reference run "<<ref_run<<")</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HB->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HB->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_distribution.gif\" alt=\"hb pedestal mean\">   </td>" << std::endl;
  if(PedestalsRmsHB->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHB->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_rms_distribution.gif\" alt=\"hb pedestal rms mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HBref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HBref->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_ref_distribution.gif\" alt=\"hb pedestal-reference mean\">   </td>" << std::endl;
  if(PedestalsRmsHBref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHBref->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_rms_ref_distribution.gif\" alt=\"hb pedestal rms-reference mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">HE Pedestal plots (Reference run "<<ref_run<<")</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HE->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0);  
  PedestalsAve4HE->Draw();
  can->SaveAs((htmlDir + "he_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_distribution.gif\" alt=\"he pedestal mean\">   </td>" << std::endl;
  if(PedestalsRmsHE->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHE->Draw();
  can->SaveAs((htmlDir + "he_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_rms_distribution.gif\" alt=\"he pedestal rms mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HEref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HEref->Draw();
  can->SaveAs((htmlDir + "he_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_ref_distribution.gif\" alt=\"he pedestal-reference mean\">   </td>" << std::endl;
  if(PedestalsRmsHEref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHEref->Draw();
  can->SaveAs((htmlDir + "he_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_rms_ref_distribution.gif\" alt=\"he pedestal rms-reference mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">HO Pedestal plots (Reference run "<<ref_run<<")</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HO->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HO->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_distribution.gif\" alt=\"ho pedestal mean\">   </td>" << std::endl;
  if(PedestalsRmsHO->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHO->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_rms_distribution.gif\" alt=\"ho pedestal rms mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  // SIMP
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4Simp->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4Simp->Draw();
  can->SaveAs((htmlDir + "sipm_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"sipm_pedestal_distribution.gif\" alt=\"sipm pedestal mean\">   </td>" << std::endl;
  if(PedestalsRmsSimp->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsSimp->Draw();
  can->SaveAs((htmlDir + "simp_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"simp_pedestal_rms_distribution.gif\" alt=\"sipm pedestal rms mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  // SIMP
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HOref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HOref->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_ref_distribution.gif\" alt=\"ho pedestal-reference mean\">   </td>" << std::endl;
  if(PedestalsRmsHOref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHOref->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_rms_ref_distribution.gif\" alt=\"ho pedestal rms-reference mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  /////////////////////////////////////////// 

  htmlFile << "<h2 align=\"center\">HF Pedestal plots (Reference run "<<ref_run<<")</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HF->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HF->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_distribution.gif\" alt=\"hf pedestal mean\">   </td>" << std::endl;
  if(PedestalsRmsHF->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHF->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_rms_distribution.gif\" alt=\"hf pedestal rms mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  if(PedestalsAve4HFref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HFref->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_ref_distribution.gif\" alt=\"hf pedestal-reference mean\">   </td>" << std::endl;
  if(PedestalsRmsHFref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHFref->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_rms_ref_distribution.gif\" alt=\"hf pedestal rms-reference mean\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();
  can->Close();
}

HcalDetDiagPedestalClient::~HcalDetDiagPedestalClient()
{}
