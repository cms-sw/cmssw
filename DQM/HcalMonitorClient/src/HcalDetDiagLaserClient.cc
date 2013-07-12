#include "DQM/HcalMonitorClient/interface/HcalDetDiagLaserClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include <iostream>

/*
 * \file HcalDetDiagLaserClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.7 $
 * \author J. Temple
 * \brief Hcal DetDiagLaser Client class
 */
typedef struct{
int eta;
int phi;
}Raddam_ch;
Raddam_ch RADDAM_CH[56]={{-30,15},{-32,15},{-34,15},{-36,15},{-38,15},{-40,15},{-41,15},
                         {-30,35},{-32,35},{-34,35},{-36,35},{-38,35},{-40,35},{-41,35},
                         {-30,51},{-32,51},{-34,51},{-36,51},{-38,51},{-40,51},{-41,51},
                         {-30,71},{-32,71},{-34,71},{-36,71},{-38,71},{-40,71},{-41,71},
                         {30, 01},{32, 01},{34, 01},{36, 01},{38, 01},{40, 71},{41, 71},
                         {30, 21},{32, 21},{34, 21},{36, 21},{38, 21},{40, 19},{41, 19},
                         {30, 37},{32, 37},{34, 37},{36, 37},{38, 37},{40, 35},{41, 35},
                         {30, 57},{32, 57},{34, 57},{36, 57},{38, 57},{40, 55},{41, 55}};
using namespace std;
using namespace edm;

HcalDetDiagLaserClient::HcalDetDiagLaserClient(std::string myname)
{
  name_=myname;   status=0;
  needLogicalMap_=true;
}

HcalDetDiagLaserClient::HcalDetDiagLaserClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<string>("DetDiagLaserFolder","DetDiagLaserMonitor_Hcal/"); // DetDiagLaserMonitor_Hcal/
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("DetDiagLaser_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("DetDiagLaser_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  
  minerrorrate_ = ps.getUntrackedParameter<double>("DetDiagLaser_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.05));
  minevents_    = ps.getUntrackedParameter<int>("DetDiagLaser_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
  needLogicalMap_=true;
}

void HcalDetDiagLaserClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalDetDiagLaserClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalDetDiagLaserClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalDetDiagLaserClient::calculateProblems()"<<std::endl;
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
  TH2F* BadTiming[4];
  TH2F* BadEnergy[4];
  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      BadTiming[i]=0;
      BadEnergy[i]=0;
      string s=subdir_+name[i]+" Problem Bad Laser Timing";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadTiming[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadTiming[i], debug_);
      else if (debug_>0) std::cout <<"<HcalDetDiagLaserClient::calculateProblems> could not get histogram '"<<s<<"'"<<std::endl;
      s=subdir_+name[i]+" Problem Bad Laser Energy";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadEnergy[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadEnergy[i], debug_);
      else if (debug_>0) std::cout <<"<HcalDetDiagLaserClient::calculateProblems> could not get histogram '"<<s<<"'"<<std::endl;
    }      

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;
    
      //totalevents=DigiPresentByDepth[d]->GetBinContent(0);
      totalevents=0;
      // Check underflow bins for events processed
      if (BadTiming[d]!=0) totalevents += BadTiming[d]->GetBinContent(0);
      if (BadEnergy[d]!=0) totalevents += BadEnergy[d]->GetBinContent(0);
      //if (totalevents==0 || totalevents<minevents_) continue;
      
      totalevents=1; // temporary value pending removal of histogram normalization from tasks

      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;
	      if (BadTiming[d]!=0) problemvalue += BadTiming[d]->GetBinContent(eta+1,phi+1)*1./totalevents;
	      if (BadEnergy[d]!=0) problemvalue += BadEnergy[d]->GetBinContent(eta+1,phi+1)*1./totalevents;
	      if (problemvalue==0) continue;
	      // problem value is a rate; we can normalize it here
	      problemvalue = min(1.,problemvalue);
	      
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
      if (debug_>0) std::cout <<"<HcalDetDiagLaserClient::calculateProblems> ProblemCells histogram does not exist!"<<endl;
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

void HcalDetDiagLaserClient::beginJob()
{
  dqmStore_ = Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalDetDiagLaserClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalDetDiagLaserClient::endJob(){}

void HcalDetDiagLaserClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalDetDiagLaserClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemDetDiagLaser",
				 " Problem DetDiagLaser Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_DetDiagLaser");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem DetDiagLaser Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;
}

void HcalDetDiagLaserClient::endRun(void){analyze();}

void HcalDetDiagLaserClient::setup(void){}
void HcalDetDiagLaserClient::cleanup(void){}

bool HcalDetDiagLaserClient::hasErrors_Temp(void)
{
   if(status&2) return true;
    return false;

  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalDetDiagLaserClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalDetDiagLaserClient::hasWarnings_Temp(void){
    if(status&1) return true;
    return false;
}
bool HcalDetDiagLaserClient::hasOther_Temp(void){return false;}
bool HcalDetDiagLaserClient::test_enabled(void){return true;}


void HcalDetDiagLaserClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

} //void HcalDetDiagLaserClient::updateChannelStatus
static void printTableHeader(ofstream& file,std::string  header){
     file << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">"<< endl;
     file << "<head>"<< endl;
     file << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>"<< endl;
     file << "<title>"<< header <<"</title>"<< endl;
     file << "<style type=\"text/css\">"<< endl;
     file << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"<< endl;
     file << "   td.s0 { font-family: arial, arial ce, helvetica; }"<< endl;
     file << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< endl;
     file << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }"<< endl;
     file << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }"<< endl;
     file << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }"<< endl;
     file << "</style>"<< endl;
     file << "<body>"<< endl;
     file << "<table>"<< endl;
}
static void printTableLine(ofstream& file,int ind,HcalDetId& detid,HcalFrontEndId& lmap_entry,HcalElectronicsId &emap_entry,std::string comment=""){
   if(ind==0){
     file << "<tr>";
     file << "<td class=\"s4\" align=\"center\">#</td>"    << endl;
     file << "<td class=\"s1\" align=\"center\">ETA</td>"  << endl;
     file << "<td class=\"s1\" align=\"center\">PHI</td>"  << endl;
     file << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< endl;
     file << "<td class=\"s1\" align=\"center\">RBX</td>"  << endl;
     file << "<td class=\"s1\" align=\"center\">RM</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">QIE</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">ADC</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">CRATE</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">DCC</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << endl;
     file << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << endl;
     if(comment[0]!=0) file << "<td class=\"s1\" align=\"center\">Comment</td>"   << endl;
     file << "</tr>"   << endl;
   }
   std::string raw_class;
   file << "<tr>"<< endl;
   if((ind%2)==1){
      raw_class="<td class=\"s2\" align=\"center\">";
   }else{
      raw_class="<td class=\"s3\" align=\"center\">";
   }
   file << "<td class=\"s4\" align=\"center\">" << ind+1 <<"</td>"<< endl;
   file << raw_class<< detid.ieta()<<"</td>"<< endl;
   file << raw_class<< detid.iphi()<<"</td>"<< endl;
   file << raw_class<< detid.depth() <<"</td>"<< endl;
   file << raw_class<< lmap_entry.rbx()<<"</td>"<< endl;
   file << raw_class<< lmap_entry.rm() <<"</td>"<< endl;
   file << raw_class<< lmap_entry.pixel()<<"</td>"<< endl;
   file << raw_class<< lmap_entry.rmFiber() <<"</td>"<< endl;
   file << raw_class<< lmap_entry.fiberChannel()<<"</td>"<< endl;
   file << raw_class<< lmap_entry.qieCard() <<"</td>"<< endl;
   file << raw_class<< lmap_entry.adc()<<"</td>"<< endl;
   file << raw_class<< emap_entry.readoutVMECrateId()<<"</td>"<< endl;
   file << raw_class<< emap_entry.dccid()<<"</td>"<< endl;
   file << raw_class<< emap_entry.spigot()<<"</td>"<< endl;
   file << raw_class<< emap_entry.fiberIndex()<<"</td>"<< endl;
   file << raw_class<< emap_entry.htrSlot()<<"</td>"<< endl;
   file << raw_class<< emap_entry.htrTopBottom()<<"</td>"<< endl;
   if(comment[0]!=0) file << raw_class<< comment<<"</td>"<< endl;
}
static void printTableTail(ofstream& file){
     file << "</table>"<< endl;
     file << "</body>"<< endl;
     file << "</html>"<< endl;
}

bool HcalDetDiagLaserClient::validHtmlOutput(){
  string s=subdir_+"HcalDetDiagLaserMonitor Event Number";
  MonitorElement *me = dqmStore_->get(s.c_str());
  int n=0;
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &n);
  }
  if(n<100) return false;
  return true;
}
void HcalDetDiagLaserClient::htmlOutput(string htmlDir){
  if(dqmStore_==0){
      if (debug_>0) std::cout <<"<HcalDetDiagLaserClient::htmlOutput> dqmStore object does not exist!"<<std::endl;
      return;
  }
  if(debug_>2) std::cout <<"\t<HcalDetDiagLaserClient::htmlOutput>  Preparing html for task: "<<name_<<std::endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////////

  HcalElectronicsMap emap=logicalMap_->generateHcalElectronicsMap();

///////////////////////////////////////////////////////////////////////////////////////////////////////////
  string ref_run,s;
  MonitorElement* me;
  TH1F *hbheEnergy=0;
  TH1F *hbheTiming=0;
  TH1F *hbheEnergyRMS=0;
  TH1F *hbheTimingRMS=0;
  TH1F *hoEnergy=0;
  TH1F *hoTiming=0;
  TH1F *hoEnergyRMS=0;
  TH1F *hoTimingRMS=0;
  TH1F *hfEnergy=0;
  TH1F *hfTiming=0;
  TH1F *hfEnergyRMS=0;
  TH1F *hfTimingRMS=0;
  TH1F *hb=0;
  TH1F *he=0;
  TH1F *ho=0;
  TH1F *hf=0; 
  TH2F *Time2Dhbhehf=0;
  TH2F *Time2Dho=0;
  TH2F *Energy2Dhbhehf=0;
  TH2F *Energy2Dho=0;
  TH2F *refTime2Dhbhehf=0;
  TH2F *refTime2Dho=0;
  TH2F *refEnergy2Dhbhehf=0;
  TH2F *refEnergy2Dho=0;
  int HBpresent_=0,HEpresent_=0,HOpresent_=0,HFpresent_=0;

 
  me=dqmStore_->get(prefixME_+"HcalInfo/HBpresent");
  if(me!=0) HBpresent_=me->getIntValue();
  me=dqmStore_->get(prefixME_+"HcalInfo/HEpresent");
  if(me!=0) HEpresent_=me->getIntValue();
  me=dqmStore_->get(prefixME_+"HcalInfo/HOpresent");
  if(me!=0) HOpresent_=me->getIntValue();
  me=dqmStore_->get(prefixME_+"HcalInfo/HFpresent");
  if(me!=0) HFpresent_=me->getIntValue();

  s=subdir_+"Summary Plots/HBHE Laser Energy Distribution"; me=dqmStore_->get(s.c_str()); 
  if(me!=0) hbheEnergy=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hbheEnergy, debug_);  else return;
  s=subdir_+"Summary Plots/HBHE Laser Timing Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hbheTiming=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hbheTiming, debug_); else return;
  s=subdir_+"Summary Plots/HBHE Laser Energy RMS_div_Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hbheEnergyRMS= HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hbheEnergyRMS, debug_); else return;
  s=subdir_+"Summary Plots/HBHE Laser Timing RMS Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hbheTimingRMS= HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hbheTimingRMS, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hoEnergy = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hoEnergy, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser Timing Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hoTiming = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hoTiming, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser Energy RMS_div_Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hoEnergyRMS = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hoEnergyRMS, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser Timing RMS Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hoTimingRMS  = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hoTimingRMS, debug_); else return;
  s=subdir_+"Summary Plots/HF Laser Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hfEnergy  = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hfEnergy, debug_); else return;
  s=subdir_+"Summary Plots/HF Laser Timing Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hfTiming  = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hfTiming, debug_); else return;
  s=subdir_+"Summary Plots/HF Laser Energy RMS_div_Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hfEnergyRMS = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hfEnergyRMS, debug_); else return;
  s=subdir_+"Summary Plots/HF Laser Timing RMS Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) hfTimingRMS = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hfTimingRMS, debug_); else return;

  s=subdir_+"Summary Plots/HB RBX average Time-Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) hb     = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hb, debug_); else return;
  s=subdir_+"Summary Plots/HE RBX average Time-Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) he     = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, he, debug_); else return;
  s=subdir_+"Summary Plots/HO RBX average Time-Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) ho     = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, ho, debug_); else return;
  s=subdir_+"Summary Plots/HF RoBox average Time-Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) hf     = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, hf, debug_); else return;

  s=subdir_+"Summary Plots/Laser Timing HBHEHF"; me=dqmStore_->get(s.c_str());
  if(me!=0) Time2Dhbhehf  = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Time2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/Laser Timing HO"; me=dqmStore_->get(s.c_str());
  if(me!=0) Time2Dho      = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Time2Dho, debug_); else return;
  s=subdir_+"Summary Plots/Laser Energy HBHEHF"; me=dqmStore_->get(s.c_str());
  if(me!=0) Energy2Dhbhehf= HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Energy2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/Laser Energy HO"; me=dqmStore_->get(s.c_str());
  if(me!=0) Energy2Dho    = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Energy2Dho, debug_); else return;
  s=subdir_+"Summary Plots/HBHEHF Laser (Timing-Ref)+1"; me=dqmStore_->get(s.c_str());
  if(me!=0) refTime2Dhbhehf  = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, refTime2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser (Timing-Ref)+1"; me=dqmStore_->get(s.c_str());
  if(me!=0) refTime2Dho      = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, refTime2Dho, debug_); else return;
  s=subdir_+"Summary Plots/HBHEHF Laser Energy_div_Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) refEnergy2Dhbhehf= HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, refEnergy2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/HO Laser Energy_div_Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) refEnergy2Dho    = HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, refEnergy2Dho, debug_); else return;

  TH1F *Raddam[56];
  char str[100];
  for(int i=0;i<56;i++){  
       sprintf(str,"RADDAM (%i %i)",RADDAM_CH[i].eta,RADDAM_CH[i].phi);
       s=subdir_+"Raddam Plots/"+str; me=dqmStore_->get(s.c_str());
       if(me!=0) Raddam[i] = HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, Raddam[i], debug_);
       Raddam[i]->SetXTitle("TS");
       Raddam[i]->SetTitle(str);
  }
  
  int ievt_ = -1,runNo=-1;
  s=subdir_+"HcalDetDiagLaserMonitor Event Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  s=subdir_+"HcalDetDiagLaserMonitor Run Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &runNo);
  }
  s=subdir_+"HcalDetDiagLaserMonitor Reference Run";
  me = dqmStore_->get(s.c_str());
  if(me) {
    string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
  }

  int  badT=0;
  int  badE=0;
  int  HBP[2]={0,0};
  int  HBM[2]={0,0};
  int  HEP[2]={0,0};
  int  HEM[2]={0,0};
  int  HFP[2]={0,0};
  int  HFM[2]={0,0};
  int  HO[2] ={0,0};
  int  newHBP[2]={0,0};
  int  newHBM[2]={0,0};
  int  newHEP[2]={0,0};
  int  newHEM[2]={0,0};
  int  newHFP[2]={0,0};
  int  newHFM[2]={0,0};
  int  newHO[2] ={0,0};

  TH2F* BadTiming_val[4];
  TH2F* BadEnergy_val[4];
  std::vector<std::string> name = HcalEtaPhiHistNames();
  for(int i=0;i<4;++i){
      BadTiming_val[i]=0;
      BadEnergy_val[i]=0;
      string s=subdir_+"Plots for client/"+name[i]+" Laser Timing difference";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadTiming_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadTiming_val[i], debug_); else return;
      s=subdir_+"Plots for client/"+name[i]+" Laser Energy difference";
      me=dqmStore_->get(s.c_str());
      if (me!=0) BadEnergy_val[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadEnergy_val[i], debug_); else return;
  }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ofstream badTiming; 
  badTiming.open((htmlDir+"bad_timing_table.html").c_str());
  printTableHeader(badTiming,"Bad Timing Channels list");
  ofstream badEnergy; 
  badEnergy.open((htmlDir+"bad_energy_table.html").c_str());
  printTableHeader(badEnergy,"Bad Energy Channels list");

 for(int d=0;d<4;++d){
      int etabins=BadTiming_val[d]->GetNbinsX();
      int phibins=BadTiming_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          HcalSubdetector subdet=HcalEmpty;
          if(isHB(eta,d+1))subdet=HcalBarrel;
	     else if (isHE(eta,d+1)) subdet=HcalEndcap;
	     else if (isHF(eta,d+1)) subdet=HcalForward;
	     else if (isHO(eta,d+1)) subdet=HcalOuter;
	  HcalDetId hcalid(subdet, ieta, phi+1, (int)(d+1));
          float val=BadTiming_val[d]->GetBinContent(eta+1,phi+1);
	  if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[0]++;}else{ HBM[0]++;} badT++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[0]++;}else{ newHBM[0]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[0]++;}else{ HEM[0]++;} badT++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[0]++;}else{ newHEM[0]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[0]++;}else{ HFM[0]++;} badT++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[0]++;}else{ newHFM[0]++;}}
            }	
            if(subdet==HcalOuter){
               HO[0]++;badT++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[0]++;}
            }	
         }
         val=BadEnergy_val[d]->GetBinContent(eta+1,phi+1);
	 if(val!=0){
            if(subdet==HcalBarrel){
               if(ieta>0){ HBP[1]++;}else{ HBM[1]++;} badE++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHBP[1]++;}else{ newHBM[1]++;}}
            }	
            if(subdet==HcalEndcap){
               if(ieta>0){ HEP[1]++;}else{ HEM[1]++;} badE++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHEP[1]++;}else{ newHEM[1]++;}}
            }	
            if(subdet==HcalForward){
               if(ieta>0){ HFP[1]++;}else{ HFM[1]++;} badE++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){if(ieta>0){ newHFP[1]++;}else{ newHFM[1]++;}}
            }	
            if(subdet==HcalOuter){
               HO[1]++;badT++;
               if(badstatusmap.find(hcalid)==badstatusmap.end()){newHO[1]++;}
            }	
        }
     }
  }


  int cnt=0;
  if((HBP[0]+HBM[0])>0){
    badTiming << "<tr><td align=\"center\"><h3>"<< "HB" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadTiming_val[d]->GetNbinsX();
      int phibins=BadTiming_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHB(eta,d+1)) continue;
          float val=BadTiming_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalBarrel,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Time-Ref=%.2f",val);
	  printTableLine(badTiming,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    }
  } 
  cnt=0;
  if((HEP[0]+HEM[0])>0){
    badTiming << "<tr><td align=\"center\"><h3>"<< "HE" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadTiming_val[d]->GetNbinsX();
      int phibins=BadTiming_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHE(eta,d+1)) continue;
          float val=BadTiming_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalEndcap,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Time-Ref=%.2f",val);
	  printTableLine(badTiming,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }
  cnt=0;
  if(HO[0]>0){
    badTiming << "<tr><td align=\"center\"><h3>"<< "HO" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadTiming_val[d]->GetNbinsX();
      int phibins=BadTiming_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHO(eta,d+1)) continue;
          float val=BadTiming_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalOuter,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Time-Ref=%.2f",val);
	  printTableLine(badTiming,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }
  cnt=0;
  if((HFP[0]+HFM[0])>0){
    badTiming << "<tr><td align=\"center\"><h3>"<< "HF" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadTiming_val[d]->GetNbinsX();
      int phibins=BadTiming_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHF(eta,d+1)) continue;
          float val=BadTiming_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalForward,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Time-Ref=%.2f",val);
	  printTableLine(badTiming,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cnt=0;
  if((HBP[1]+HBM[1])>0){
    badEnergy << "<tr><td align=\"center\"><h3>"<< "HB" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadEnergy_val[d]->GetNbinsX();
      int phibins=BadEnergy_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHB(eta,d+1)) continue;
          float val=BadEnergy_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalBarrel,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Energy/Ref=%.2f",val);
	  printTableLine(badEnergy,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    }
  } 
  cnt=0;
  if((HEP[1]+HEM[1])>0){
    badEnergy << "<tr><td align=\"center\"><h3>"<< "HE" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadEnergy_val[d]->GetNbinsX();
      int phibins=BadEnergy_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHE(eta,d+1)) continue;
          float val=BadEnergy_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalEndcap,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Energy/Ref=%.2f",val);
	  printTableLine(badEnergy,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }
  cnt=0;
  if(HO[1]>0){
    badEnergy << "<tr><td align=\"center\"><h3>"<< "HO" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadEnergy_val[d]->GetNbinsX();
      int phibins=BadEnergy_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHO(eta,d+1)) continue;
          float val=BadEnergy_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalOuter,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Energy/Ref=%.2f",val);
	  printTableLine(badEnergy,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }
  cnt=0;
  if((HFP[1]+HFM[1])>0){
    badEnergy << "<tr><td align=\"center\"><h3>"<< "HF" <<"</h3></td></tr>" << endl;
    for(int d=0;d<4;++d){
      int etabins=BadEnergy_val[d]->GetNbinsX();
      int phibins=BadEnergy_val[d]->GetNbinsY();
      for(int phi=0;phi<phibins;++phi)for(int eta=0;eta<etabins;++eta){
	  int ieta=CalcIeta(eta,d+1);
	  if(ieta==-9999) continue;
          if(!isHF(eta,d+1)) continue;
          float val=BadEnergy_val[d]->GetBinContent(eta+1,phi+1);
	  if(val==0) continue;
          HcalDetId hcalid(HcalForward,ieta,phi+1,d+1);
	  HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(hcalid);
	  HcalElectronicsId emap_entry=emap.lookup(hcalid);
          sprintf(str,"Energy/Ref=%.2f",val);
	  printTableLine(badEnergy,cnt++,hcalid,lmap_entry,emap_entry,str);
      } 
    } 
  }

  printTableTail(badTiming);
  badTiming.close();
  printTableTail(badEnergy);
  badEnergy.close();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ofstream htmlFile;
  string outfile=htmlDir+name_+".html";
  htmlFile.open(outfile.c_str());
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
  TCanvas *can=new TCanvas("HcalDetDiagLaserClient","HcalDetDiagLaserClient",0,0,500,350);
  can->cd();
  
  if(Raddam[0]->GetEntries()>0){
     ofstream RADDAM;
     RADDAM.open((htmlDir + "RADDAM_"+name_).c_str());
     RADDAM << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">"<< endl;
     RADDAM << "<head>"<< endl;
     RADDAM << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>"<< endl;
     RADDAM << "<title>"<< "RADDAM channels" <<"</title>"<< endl;
     RADDAM << "<style type=\"text/css\">"<< endl;
     RADDAM << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"<< endl;
     RADDAM << "   td.s0 { font-family: arial, arial ce, helvetica; }"<< endl;
     RADDAM << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< endl;
     RADDAM << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }"<< endl;
     RADDAM << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }"<< endl;
     RADDAM << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }"<< endl;
     RADDAM << "</style>"<< endl;
     RADDAM << "<body>"<< endl;
     RADDAM << "<h2>Run "<< runNo<<": RADDAM channels event shape </h2>" << endl;
     RADDAM << "<table>"<< endl;

     char str[100];
     for(int i=0;i<28;i++){
         RADDAM << "<tr align=\"left\">" << endl;
         //Raddam[2*i]->SetStats(0);
         //Raddam[2*i+1]->SetStats(0);
         Raddam[2*i]->Draw();    sprintf(str,"%02d",2*i);    can->SaveAs((htmlDir + "raddam_ch"+str+".gif").c_str());
         Raddam[2*i+1]->Draw();  sprintf(str,"%02d",2*i+1);  can->SaveAs((htmlDir + "raddam_ch"+str+".gif").c_str());
	 sprintf(str,"raddam_ch%02d.gif",2*i);
         RADDAM << "<td align=\"center\"><img src=\""<<str<<"\" alt=\"raddam channel\">   </td>" << endl;
	 sprintf(str,"raddam_ch%02d.gif",2*i+1);
         RADDAM << "<td align=\"center\"><img src=\""<<str<<"\" alt=\"raddam channel\">   </td>" << endl;
         RADDAM << "</tr>" << endl;
     }

     RADDAM << "</table>"<< endl;
     RADDAM << "</body>"<< endl;
     RADDAM << "</html>"<< endl;
     RADDAM.close();
  }

  Time2Dhbhehf->SetXTitle("i#eta");
  Time2Dhbhehf->SetYTitle("i#phi");
  Time2Dho->SetXTitle("i#eta");
  Time2Dho->SetYTitle("i#phi");
  Energy2Dhbhehf->SetXTitle("i#eta");
  Energy2Dhbhehf->SetYTitle("i#phi");
  Energy2Dho->SetXTitle("i#eta");
  Energy2Dho->SetYTitle("i#phi");
  refTime2Dhbhehf->SetXTitle("i#eta");
  refTime2Dhbhehf->SetYTitle("i#phi");
  refTime2Dho->SetXTitle("i#eta");
  refTime2Dho->SetYTitle("i#phi");
  refEnergy2Dhbhehf->SetXTitle("i#eta");
  refEnergy2Dhbhehf->SetYTitle("i#phi");
  refEnergy2Dho->SetXTitle("i#eta");
  refEnergy2Dho->SetYTitle("i#phi");
  refTime2Dhbhehf->SetMinimum(0);
  refTime2Dhbhehf->SetMaximum(2);
  refTime2Dho->SetMinimum(0);
  refTime2Dho->SetMaximum(2);
  refEnergy2Dhbhehf->SetMinimum(0.5);
  refEnergy2Dhbhehf->SetMaximum(1.5);
  refEnergy2Dho->SetMinimum(0.5);
  refEnergy2Dho->SetMaximum(1.5);
  
  Time2Dhbhehf->SetNdivisions(36,"Y");
  Time2Dho->SetNdivisions(36,"Y");
  Energy2Dhbhehf->SetNdivisions(36,"Y");
  Energy2Dho->SetNdivisions(36,"Y");
  refTime2Dhbhehf->SetNdivisions(36,"Y");
  refTime2Dho->SetNdivisions(36,"Y");
  refEnergy2Dhbhehf->SetNdivisions(36,"Y");
  refEnergy2Dho->SetNdivisions(36,"Y");

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Detector Diagnostics Laser Monitor</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
 
  htmlFile << "<style type=\"text/css\">"<< endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FF7700; text-align: center;}"<< endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: red; }"<< endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: yellow; }"<< endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: green; }"<< endl;
  htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: silver; }"<< endl;
  std::string state[4]={"<td class=\"s2\" align=\"center\">",
			"<td class=\"s3\" align=\"center\">",
			"<td class=\"s4\" align=\"center\">",
			"<td class=\"s5\" align=\"center\">"};
  htmlFile << "</style>"<< endl;
 
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics Laser Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
 
  htmlFile << "<table width=100% border=1>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">SebDet</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad Timing</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad Energy</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  int ind1=0,ind2=0;
  htmlFile << "<td class=\"s1\" align=\"center\">HB+</td>" << endl;
  if(HBP[0]==0) ind1=2; if(HBP[0]>0 && HBP[0]<=12) ind1=1; if(HBP[0]>12) ind1=0; 
  if(HBP[1]==0) ind2=2; if(HBP[1]>0 && HBP[1]<=12) ind2=1; if(HBP[1]>12) ind2=0; 
  if(!HBpresent_) ind1=ind2=3;  
  htmlFile << state[ind1] << HBP[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HBP[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HB-</td>" << endl;
  if(HBM[0]==0) ind1=2; if(HBM[0]>0 && HBP[0]<=12) ind1=1; if(HBM[0]>12) ind1=0; 
  if(HBM[1]==0) ind2=2; if(HBM[1]>0 && HBP[1]<=12) ind2=1; if(HBM[1]>12) ind2=0; 
  if(!HBpresent_) ind1=ind2=3;  
  htmlFile << state[ind1] << HBM[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HBM[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE+</td>" << endl;
  if(HEP[0]==0) ind1=2; if(HEP[0]>0 && HEP[0]<=12) ind1=1; if(HEP[0]>12) ind1=0; 
  if(HEP[1]==0) ind2=2; if(HEP[1]>0 && HEP[1]<=12) ind2=1; if(HEP[1]>12) ind2=0;
  if(!HEpresent_) ind1=ind2=3;  
  if(ind1==0 || ind2==0) status|=2; else if(ind1==1 || ind2==1) status|=1;
  htmlFile << state[ind1] << HEP[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HEP[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE-</td>" << endl;
  if(HEM[0]==0) ind1=2; if(HEM[0]>0 && HEM[0]<=12) ind1=1; if(HEM[0]>12) ind1=0; 
  if(HEM[1]==0) ind2=2; if(HEM[1]>0 && HEM[1]<=12) ind2=1; if(HEM[1]>12) ind2=0; 
  if(!HEpresent_) ind1=ind2=3;  
  if(ind1==0 || ind2==0) status|=2; else if(ind1==1 || ind2==1) status|=1;
  htmlFile << state[ind1] << HEM[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HEM[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF+</td>" << endl;
  if(HFP[0]==0) ind1=2; if(HFP[0]>0 && HFP[0]<=12) ind1=1; if(HFP[0]>12) ind1=0; 
  if(HFP[1]==0) ind2=2; if(HFP[1]>0 && HFP[1]<=12) ind2=1; if(HFP[1]>12) ind2=0; 
  if(!HOpresent_) ind1=ind2=3;  
  if(ind1==0 || ind2==0) status|=2; else if(ind1==1 || ind2==1) status|=1;
  htmlFile << state[ind1] << HFP[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HFP[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF-</td>" << endl;
  if(HFM[0]==0) ind1=2; if(HFM[0]>0 && HFM[0]<=12) ind1=1; if(HFM[0]>12) ind1=0; 
  if(HFM[1]==0) ind2=2; if(HFM[1]>0 && HFM[1]<=12) ind2=1; if(HFM[1]>12) ind2=0; 
  if(!HFpresent_) ind1=ind2=3;  
  if(ind1==0 || ind2==0) status|=2; else if(ind1==1 || ind2==1) status|=1;
  htmlFile << state[ind1] << HFM[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HFM[1] <<"</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HO</td>" << endl;
  if(HO[0]==0) ind1=2; if(HO[0]>0 && HO[0]<=12) ind1=1; if(HO[0]>12) ind1=0; 
  if(HO[1]==0) ind2=2; if(HO[1]>0 && HO[1]<=12) ind2=1; if(HO[1]>12) ind2=0; 
  if(!HFpresent_) ind1=ind2=3;  
  if(ind1==0 || ind2==0) status|=2; else if(ind1==1 || ind2==1) status|=1;
  htmlFile << state[ind1] << HO[0] <<"</td>" << endl;
  htmlFile << state[ind2] << HO[1] <<"</td>" << endl;

  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  if((badT+badE)>0){
      htmlFile << "<table width=100% border=1><tr>" << endl;
      if(badT>0)  htmlFile << "<td><a href=\"" << "bad_timing_table.html" <<"\">list of bad timing channels</a></td>";
      if(badE>0) htmlFile << "<td><a href=\"" << "bad_energy_table.html" <<"\">list of bad energy channels</a></td>";
      htmlFile << "</tr></table>" << endl;
  }

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
  if(Raddam[0]->GetEntries()>0){
    htmlFile << "<h2 align=\"center\"><a href=\"" << ("RADDAM_"+name_).c_str() <<"\">RADDAM channels</a><h2>";
    htmlFile << "<hr>" << endl;
  }
  
  htmlFile << "<h2 align=\"center\">Stability Laser plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  
  can->SetGridy();
  can->SetGridx();
  
  htmlFile << "<tr align=\"left\">" << endl;
  refTime2Dhbhehf->SetStats(0);
  refTime2Dho->SetStats(0);
  refTime2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "ref_laser_timing_hbhehf.gif").c_str());
  refTime2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "ref_laser_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_timing_hbhehf.gif\" alt=\"ref laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_timing_ho.gif\" alt=\"ref laser timing distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  
  if(hb!=0 && he!=0 && ho!=0 && hf!=0){
  hb->SetMarkerStyle(22);
  hb->SetMarkerColor(kRed);
  hb->GetYaxis()->SetRangeUser(hb->GetMinimum()-1,hb->GetMaximum()+1);
  hb->GetXaxis()->SetNdivisions(520);
  
  he->SetMarkerStyle(22);
  he->SetMarkerColor(kRed);
  he->GetYaxis()->SetRangeUser(he->GetMinimum()-1,he->GetMaximum()+1);
  he->GetXaxis()->SetNdivisions(520);
  
  ho->SetMarkerStyle(22);
  ho->SetMarkerColor(kRed);
  ho->GetYaxis()->SetRangeUser(ho->GetMinimum()-1,ho->GetMaximum()+1);
  ho->GetXaxis()->SetNdivisions(520);
  
  hf->SetMarkerStyle(22);
  hf->SetMarkerColor(kRed);
  hf->GetYaxis()->SetRangeUser(hf->GetMinimum()-1,hf->GetMaximum()+1);
  hf->GetXaxis()->SetNdivisions(520);  
  hb->SetStats(0);
  he->SetStats(0);
  ho->SetStats(0);
  hf->SetStats(0);
 
  
  hb->GetXaxis()->SetBit(TAxis::kLabelsVert);
  he->GetXaxis()->SetBit(TAxis::kLabelsVert);
  ho->GetXaxis()->SetBit(TAxis::kLabelsVert);
  hf->GetXaxis()->SetBit(TAxis::kLabelsVert);
  hb->GetXaxis()->SetLabelSize(0.05);
  he->GetXaxis()->SetLabelSize(0.05);
  ho->GetXaxis()->SetLabelSize(0.05);
  hf->GetXaxis()->SetLabelSize(0.05);
 
  
  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hb_rbx_timing1D.gif\" alt=\"rbx timing\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"he_rbx_timing1D.gif\" alt=\"rbx timing\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ho_rbx_timing1D.gif\" alt=\"rbx timing\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hf_rbx_timing1D.gif\" alt=\"rbx timing\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  }else printf("Error\n");
  
  
  
  htmlFile << "<tr align=\"left\">" << endl;
  refEnergy2Dhbhehf->SetStats(0);
  refEnergy2Dho->SetStats(0);
  refEnergy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "ref_laser_energy_hbhehf.gif").c_str());
  refEnergy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "ref_laser_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_energy_hbhehf.gif\" alt=\"ref laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_energy_ho.gif\" alt=\"ref laser energy distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "</table>" << endl;
  
  
  htmlFile << "<h2 align=\"center\">Summary Laser plots</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Time2Dhbhehf->SetStats(0);
  Time2Dho->SetStats(0);
  Time2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "laser_timing_hbhehf.gif").c_str());
  Time2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "laser_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_hbhehf.gif\" alt=\"laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_ho.gif\" alt=\"laser timing distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  Energy2Dhbhehf->SetStats(0);
  Energy2Dho->SetStats(0);
  Energy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "laser_energy_hbhehf.gif").c_str());
  Energy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "laser_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_hbhehf.gif\" alt=\"laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_ho.gif\" alt=\"laser energy distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  
  htmlFile << "<tr align=\"left\">" << endl;  
  hbheEnergy->Draw();    can->SaveAs((htmlDir + "hbhe_laser_energy_distribution.gif").c_str());
  hbheEnergyRMS->Draw(); can->SaveAs((htmlDir + "hbhe_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_energy_distribution.gif\" alt=\"hbhe laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_energy_rms_distribution.gif\" alt=\"hbhelaser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hbheTiming->Draw();    can->SaveAs((htmlDir + "hbhe_laser_timing_distribution.gif").c_str());
  hbheTimingRMS->Draw(); can->SaveAs((htmlDir + "hbhe_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_timing_distribution.gif\" alt=\"hbhe laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_timing_rms_distribution.gif\" alt=\"hbhe laser timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;  
  hoEnergy->Draw();    can->SaveAs((htmlDir + "ho_laser_energy_distribution.gif").c_str());
  hoEnergyRMS->Draw(); can->SaveAs((htmlDir + "ho_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_energy_distribution.gif\" alt=\"ho laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_energy_rms_distribution.gif\" alt=\"ho laser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hoTiming->Draw();    can->SaveAs((htmlDir + "ho_laser_timing_distribution.gif").c_str());
  hoTimingRMS->Draw(); can->SaveAs((htmlDir + "ho_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_timing_distribution.gif\" alt=\"ho laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_timing_rms_distribution.gif\" alt=\"ho laser timing rms distribution\">   </td>" << endl;
  
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;  
  hfEnergy->Draw();    can->SaveAs((htmlDir + "hf_laser_energy_distribution.gif").c_str());
  hfEnergyRMS->Draw(); can->SaveAs((htmlDir + "hf_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_energy_distribution.gif\" alt=\"hf laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_energy_rms_distribution.gif\" alt=\"hf laser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hfTiming->Draw();    can->SaveAs((htmlDir + "hf_laser_timing_distribution.gif").c_str());
  hfTimingRMS->Draw(); can->SaveAs((htmlDir + "hf_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_timing_distribution.gif\" alt=\"hf laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_timing_rms_distribution.gif\" alt=\"hf laser timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  can->SetBottomMargin(0.2);  
  if(hb->GetEntries()>0)hb->Draw("P");else hb->Draw(); can->SaveAs((htmlDir + "hb_rbx_timing1D.gif").c_str());
  if(he->GetEntries()>0)he->Draw("P");else he->Draw(); can->SaveAs((htmlDir + "he_rbx_timing1D.gif").c_str());
  if(ho->GetEntries()>0)ho->Draw("P");else ho->Draw(); can->SaveAs((htmlDir + "ho_rbx_timing1D.gif").c_str());
  if(hf->GetEntries()>0)hf->Draw("P");else hf->Draw(); can->SaveAs((htmlDir + "hf_rbx_timing1D.gif").c_str());  

  htmlFile << "</table>" << endl;

  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  can->Close();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  htmlFile.close();
  return;
}

HcalDetDiagLaserClient::~HcalDetDiagLaserClient()
{}
