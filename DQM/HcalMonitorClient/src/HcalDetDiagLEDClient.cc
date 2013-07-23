#include "DQM/HcalMonitorClient/interface/HcalDetDiagLEDClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include <iostream>

/*
 * \file HcalDetDiagLEDClient.cc
 * 
 * $Date: 2012/11/01 11:10:07 $
 * $Revision: 1.10 $
 * \author J. Temple
 * \brief Hcal DetDiagLED Client class
 */

HcalDetDiagLEDClient::HcalDetDiagLEDClient(std::string myname){
  name_=myname;  status=0;
  needLogicalMap_=true;
}

HcalDetDiagLEDClient::HcalDetDiagLEDClient(std::string myname, const edm::ParameterSet& ps){
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("DetDiagLEDFolder","DetDiagLEDMonitor_Hcal/");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("DetDiagLED_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("DetDiagLED_BadChannelStatusMask",
                            ps.getUntrackedParameter<int>("BadChannelStatusMask",(1<<HcalChannelStatus::HcalCellDead)));
  needLogicalMap_=true;
}

void HcalDetDiagLEDClient::analyze(){
  if (debug_>2) std::cout <<"\tHcalDetDiagLEDClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalDetDiagLEDClient::calculateProblems(){}
void HcalDetDiagLEDClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual){
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

}

void HcalDetDiagLEDClient::beginJob(){
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0){
    std::cout <<"<HcalDetDiagLEDClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
    dqmStore_->showDirStructure();
  }
}
void HcalDetDiagLEDClient::endJob(){}

void HcalDetDiagLEDClient::beginRun(void){
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalDetDiagLEDClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemDetDiagLED",
				 " Problem DetDiagLED Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_DetDiagLED");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem DetDiagLED Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;
}
void HcalDetDiagLEDClient::endRun(void){analyze();}
void HcalDetDiagLEDClient::setup(void){}
void HcalDetDiagLEDClient::cleanup(void){}

bool HcalDetDiagLEDClient::hasErrors_Temp(void){
    if(status&2) return true;
    return false;
}
bool HcalDetDiagLEDClient::hasWarnings_Temp(void){
   if(status&1) return true;
   return false;
}
bool HcalDetDiagLEDClient::hasOther_Temp(void){return false;}
bool HcalDetDiagLEDClient::test_enabled(void){return true;}

bool HcalDetDiagLEDClient::validHtmlOutput(){
  std::string s=subdir_+"HcalDetDiagLEDMonitor Event Number";
  MonitorElement *me = dqmStore_->get(s.c_str());
  int n=0;
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &n);
  }
  if(n<100) return false;
  return true;
}
static void printTableHeader(ofstream& file,std::string header){
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
static void printTableLine(ofstream& file,int ind,HcalDetId& detid,HcalFrontEndId& lmap_entry,HcalElectronicsId &emap_entry, std::string comment=""){
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
double HcalDetDiagLEDClient::get_channel_status(std::string subdet,int eta,int phi,int depth,int type){
   int subdetval=-1;
   if (subdet.compare("HB")==0) subdetval=(int)HcalBarrel;
   else if (subdet.compare("HE")==0) subdetval=(int)HcalEndcap;
   else if (subdet.compare("HO")==0) subdetval=(int)HcalOuter;
   else if (subdet.compare("HF")==0) subdetval=(int)HcalForward;
   else return -1.0;
   int ietabin=CalcEtaBin(subdetval, eta, depth)+1;
   if(type==1) return ChannelStatusMissingChannels[depth-1]->GetBinContent(ietabin,phi);
   if(type==2) return ChannelStatusUnstableChannels[depth-1]->GetBinContent(ietabin,phi);
   if(type==3) return ChannelStatusUnstableLEDsignal[depth-1]->GetBinContent(ietabin,phi);
   if(type==4) return ChannelStatusLEDMean[depth-1]->GetBinContent(ietabin,phi);
   if(type==5) return ChannelStatusLEDRMS[depth-1]->GetBinContent(ietabin,phi);
   if(type==6) return ChannelStatusTimeMean[depth-1]->GetBinContent(ietabin,phi);
   if(type==7) return ChannelStatusTimeRMS[depth-1]->GetBinContent(ietabin,phi);
   return -1.0;
}
double HcalDetDiagLEDClient::get_energy(std::string subdet,int eta,int phi,int depth,int type){
   int subdetval=-1;
   if (subdet.compare("HB")==0) subdetval=(int)HcalBarrel;
   else if (subdet.compare("HE")==0) subdetval=(int)HcalEndcap;
   else if (subdet.compare("HO")==0) subdetval=(int)HcalOuter;
   else if (subdet.compare("HF")==0) subdetval=(int)HcalForward;
   else return -1.0;
   int ietabin=CalcEtaBin(subdetval, eta, depth)+1;
   if(type==1) return ChannelsLEDEnergy[depth-1]->GetBinContent(ietabin,phi);
   if(type==2) return ChannelsLEDEnergyRef[depth-1]->GetBinContent(ietabin,phi);
   return -1.0;
}
void HcalDetDiagLEDClient::htmlOutput(std::string htmlDir){
MonitorElement* me;
int  MissingCnt=0;
int  UnstableCnt=0;
int  UnstableLEDCnt=0;
int  BadTimingCnt=0;
int  HBP[7]={0,0,0,0,0,0,0},newHBP[7]={0,0,0,0,0,0,0};  
int  HBM[7]={0,0,0,0,0,0,0},newHBM[7]={0,0,0,0,0,0,0};  
int  HEP[7]={0,0,0,0,0,0,0},newHEP[7]={0,0,0,0,0,0,0}; 
int  HEM[7]={0,0,0,0,0,0,0},newHEM[7]={0,0,0,0,0,0,0};  
int  HFP[7]={0,0,0,0,0,0,0},newHFP[7]={0,0,0,0,0,0,0}; 
int  HFM[7]={0,0,0,0,0,0,0},newHFM[7]={0,0,0,0,0,0,0}; 
int  HO[7] ={0,0,0,0,0,0,0},newHO[7] ={0,0,0,0,0,0,0};  
std::string subdet[4]={"HB","HE","HO","HF"};

   if (debug_>0) std::cout << "<HcalDetDiagLEDClient::htmlOutput> Preparing  html output ..." << std::endl;
   if(!dqmStore_) return;
   HcalElectronicsMap emap=logicalMap_->generateHcalElectronicsMap();
   std::vector<std::string> name = HcalEtaPhiHistNames();

   for(int i=0;i<4;++i){
      ChannelsLEDEnergy[i]=ChannelsLEDEnergyRef[i]=ChannelStatusMissingChannels[i]=ChannelStatusUnstableChannels[i]=0;
      ChannelStatusUnstableLEDsignal[i]=ChannelStatusLEDMean[i]=ChannelStatusLEDRMS[i]=ChannelStatusTimeMean[i]=0;
      ChannelStatusTimeRMS[i]=0;
      std::string s;
      s=subdir_+"channel status/"+name[i]+" Missing Channels";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusMissingChannels[i]=HcalUtilsClient::getHisto<TH2F*>(me,cloneME_,ChannelStatusMissingChannels[i],debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" Unstable Channels";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusUnstableChannels[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusUnstableChannels[i], debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" Unstable LED";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusUnstableLEDsignal[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusUnstableLEDsignal[i], debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" LED Mean";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusLEDMean[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusLEDMean[i], debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" LED RMS";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusLEDRMS[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusLEDRMS[i], debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" Time Mean";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusTimeMean[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusTimeMean[i], debug_); else return;  
      s=subdir_+"channel status/"+name[i]+" Time RMS";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelStatusTimeRMS[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelStatusTimeRMS[i], debug_); else return;  
      s=subdir_+"Summary Plots/"+name[i]+" Channel LED Energy";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelsLEDEnergy[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelsLEDEnergy[i], debug_); else return;  
      s=subdir_+"Summary Plots/"+name[i]+" Channel LED Energy Reference";
      me=dqmStore_->get(s.c_str());
      if (me!=0) ChannelsLEDEnergyRef[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, ChannelsLEDEnergyRef[i], debug_); else return;
  }

  // check how many problems we have:
  for(int sd=0;sd<4;sd++){
     int feta=0,teta=0,fdepth=0,tdepth=0; 
     if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2; } 
     if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3; } 
     if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4; } 
     if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2; } 
     for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
        if(sd==3 && eta>-29 && eta<29) continue;
        double problem[7]={0,0,0,0,0,0,0}; 
        for(int i=0;i<6;i++){
	   problem[i] =get_channel_status(subdet[sd],eta,phi,depth,i+1);
           if(problem[i]!=0){
              HcalSubdetector SD=HcalEmpty;
              if(sd==0)SD=HcalBarrel;
	      else if(sd==1) SD=HcalEndcap;
	      else if(sd==2) SD=HcalOuter;
	      else if(sd==3) SD=HcalForward;
	      HcalDetId hcalid(SD, eta, phi, depth);
	      if(sd==0){  if(eta>0){ 
                  HBP[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHBP[i]++;
              }else{ 
                  HBM[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHBM[i]++;
              }} 
	      if(sd==1){  if(eta>0){ 
                  HEP[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHEP[i]++;
              }else{ 
                  HEM[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHEM[i]++;
              }}
	      if(sd==2){
                  HO[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHO[i]++;
              }
	      if(sd==3){  if(eta>0){ 
                  HFP[i]++;
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHFP[i]++;
              }else{
                  HFM[i]++; 
                  if(badstatusmap.find(hcalid)==badstatusmap.end())newHFM[i]++;
              }}
           }
        }
     }
  }
  // missing channels list
  ofstream Missing;
  Missing.open((htmlDir + "Missing.html").c_str());
  printTableHeader(Missing,"Missing Channels list");
  // Bad timing channels list
  ofstream BadTiming;
  BadTiming.open((htmlDir + "BadTiming.html").c_str());
  printTableHeader(BadTiming,"Bad Timing Channels list");
  // unstable channels list
  ofstream Unstable;
  Unstable.open((htmlDir + "Unstable.html").c_str());
  printTableHeader(Unstable,"Low LED signal Channels list");
  // unstable LED signal list
  ofstream BadLED;
  BadLED.open((htmlDir + "UnstableLED.html").c_str());
  printTableHeader(BadLED,"Unstable LED signal channels list");

  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && ((HBM[0]+HBP[0])==0 || (HBM[0]+HBP[0])==(1296*2))) continue;
      if(sd==1 && ((HEM[0]+HEP[0])==0 || (HEM[0]+HEP[0])==(1296*2))) continue;
      if(sd==2 && ((HO[0])==0 || HO[0]==2160))                      continue;
      if(sd==3 && ((HFM[0]+HFP[0])==0 || (HFM[0]+HFP[0])==(864*2))) continue;
      Missing << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << std::endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2; if(HBM[0]==1296) feta=0; if(HBP[0]==1296) teta=0;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3; if(HEM[0]==1296) feta=0; if(HEP[0]==1296) teta=0;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4; if(HO[0] ==2160) {feta=0; teta=0; }} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2; if(HFM[0]==864)  feta=0; if(HFP[0]==864)  teta=0; } 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double missing =get_channel_status(subdet[sd],eta,phi,depth,1);
         if(missing>0){
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
               std::string s=" ";
               if(badstatusmap.find(*detid)!=badstatusmap.end()){ s="Known problem"; }	
	       printTableLine(Missing,cnt++,*detid,lmap_entry,emap_entry,s); MissingCnt++;
	       delete detid;
         }
      }	
  }

  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[5]+HBP[5])==0) continue;
      if(sd==1 && (HEM[5]+HEP[5])==0) continue;
      if(sd==2 && (HO[5])==0)         continue;
      if(sd==3 && (HFM[5]+HFP[5])==0) continue;
      BadTiming << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << std::endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2; if(HBM[0]==1296) feta=0; if(HBP[0]==1296) teta=0;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3; if(HEM[0]==1296) feta=0; if(HEP[0]==1296) teta=0;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4; if(HO[0] ==2160) {feta=0; teta=0; }} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2; if(HFM[0]==864)  feta=0; if(HFP[0]==864)  teta=0; } 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double badtiming =get_channel_status(subdet[sd],eta,phi,depth,6);
         if(badtiming!=0){
            try{
	       char comment[100]; sprintf(comment,"Time-mean=%.1f\n",badtiming);
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(BadTiming,cnt++,*detid,lmap_entry,emap_entry,comment); BadTimingCnt++;
	       delete detid;
	    }catch(cms::Exception &){ continue;}
         }
      }	
  }
  
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[1]+HBP[1])==0) continue;
      if(sd==1 && (HEM[1]+HEP[1])==0) continue;
      if(sd==2 && (HO[1])==0)         continue;
      if(sd==3 && (HFM[1]+HFP[1])==0) continue;
      Unstable << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << std::endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4;} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2;} 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double unstable =get_channel_status(subdet[sd],eta,phi,depth,2);
         if(unstable>0){
            try{
	       char comment[100]; sprintf(comment,"%.3f%%\n",(1.0-unstable)*100.0);
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(Unstable,cnt++,*detid,lmap_entry,emap_entry,comment); UnstableCnt++;
	       delete detid;
	    }catch(cms::Exception &){ continue;}
         }
      }	
  }
  
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[2]+HBP[2])==0) continue;
      if(sd==1 && (HEM[2]+HEP[2])==0) continue;
      if(sd==2 &&  (HO[2])==0)        continue;
      if(sd==3 && (HFM[2]+HFP[2])==0) continue;
      BadLED << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << std::endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4;} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2;} 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double badled =get_channel_status(subdet[sd],eta,phi,depth,3);
         if(badled>0){
            try{
	       char comment[100]; sprintf(comment,"%.3f%%\n",(badled)*100.0);
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=logicalMap_->getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
               std::string s=" ";
               if(badstatusmap.find(*detid)!=badstatusmap.end()){ s="Known problem"; }	
	       printTableLine(BadLED,cnt++,*detid,lmap_entry,emap_entry,s); UnstableLEDCnt++;
	       delete detid;
	    }catch(cms::Exception &){ continue;}
         }
      }	
  }
  printTableTail(Missing);
  Missing.close();
  printTableTail(BadTiming);
  BadTiming.close();
  printTableTail(Unstable);
  Unstable.close();
  printTableTail(BadLED);
  BadLED.close();
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int ievt_ = -1,runNo=-1;
  std::string ref_run;
  std::string s=subdir_+"HcalDetDiagLEDMonitor Event Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  s=subdir_+"HcalDetDiagLEDMonitor Run Number";
  me = dqmStore_->get(s.c_str());
  if ( me ) {
    s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &runNo);
  } 
  s=subdir_+"HcalDetDiagLEDMonitor Reference Run";
  me = dqmStore_->get(s.c_str());
  if(me) {
    std::string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
  }
  TH1F *Energy=0,*Timing=0,*EnergyHF=0,*TimingHF=0,*EnergyRMS=0,*TimingRMS=0,*EnergyRMSHF=0,*TimingRMSHF=0;
  TH2F *Time2Dhbhehf=0,*Time2Dho=0,*Energy2Dhbhehf=0,*Energy2Dho=0;
  TH2F *HBPphi=0,*HBMphi=0,*HEPphi=0,*HEMphi=0,*HFPphi=0,*HFMphi=0,*HO0phi=0,*HO1Pphi=0,*HO2Pphi=0,*HO1Mphi=0,*HO2Mphi=0;

  s=subdir_+"Summary Plots/HBHEHO LED Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) Energy=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, Energy, debug_); else return;
  s=subdir_+"Summary Plots/HBHEHO LED Timing Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) Timing=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, Timing, debug_); else return;
  s=subdir_+"Summary Plots/HBHEHO LED Energy RMS_div_Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) EnergyRMS=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, EnergyRMS, debug_); else return;
  s=subdir_+"Summary Plots/HBHEHO LED Timing RMS Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) TimingRMS=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, TimingRMS, debug_); else return;
  s=subdir_+"Summary Plots/HF LED Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) EnergyHF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, EnergyHF, debug_); else return;
  s=subdir_+"Summary Plots/HF LED Timing Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) TimingHF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, TimingHF, debug_); else return;
  s=subdir_+"Summary Plots/HF LED Energy RMS_div_Energy Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) EnergyRMSHF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, EnergyRMSHF, debug_); else return;
  s=subdir_+"Summary Plots/HF LED Timing RMS Distribution"; me=dqmStore_->get(s.c_str());
  if(me!=0) TimingRMSHF=HcalUtilsClient::getHisto<TH1F*>(me, cloneME_, TimingRMSHF, debug_); else return;

  s=subdir_+"Summary Plots/LED Timing HBHEHF"; me=dqmStore_->get(s.c_str());
  if(me!=0) Time2Dhbhehf=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Time2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/LED Timing HO"; me=dqmStore_->get(s.c_str());
  if(me!=0) Time2Dho=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Time2Dho, debug_); else return;
  s=subdir_+"Summary Plots/LED Energy HBHEHF"; me=dqmStore_->get(s.c_str());
  if(me!=0) Energy2Dhbhehf=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Energy2Dhbhehf, debug_); else return;
  s=subdir_+"Summary Plots/LED Energy HO"; me=dqmStore_->get(s.c_str());
  if(me!=0) Energy2Dho=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, Energy2Dho, debug_); else return;

  s=subdir_+"Summary Plots/HBP Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HBPphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HBPphi, debug_); else return;
  s=subdir_+"Summary Plots/HBM Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HBMphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HBMphi, debug_); else return;
  s=subdir_+"Summary Plots/HEP Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HEPphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HEPphi, debug_); else return;
  s=subdir_+"Summary Plots/HEM Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HEMphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HEMphi, debug_); else return;
  s=subdir_+"Summary Plots/HFP Average over RM LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HFPphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HFPphi, debug_); else return;
  s=subdir_+"Summary Plots/HFM Average over RM LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HFMphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HFMphi, debug_); else return;

  s=subdir_+"Summary Plots/HO0 Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HO0phi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HO0phi, debug_); else return;
  s=subdir_+"Summary Plots/HO1P Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HO1Pphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HO1Pphi, debug_); else return;
  s=subdir_+"Summary Plots/HO2P Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HO2Pphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HO2Pphi, debug_); else return;
  s=subdir_+"Summary Plots/HO1M Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HO1Mphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HO1Mphi, debug_); else return;
  s=subdir_+"Summary Plots/HO2M Average over HPD LED Ref"; me=dqmStore_->get(s.c_str());
  if(me!=0) HO2Mphi=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HO2Mphi, debug_); else return;

  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
 
  TCanvas *can=new TCanvas("HcalDetDiagLEDClient","HcalDetDiagLEDClient",0,0,500,350);
  can->SetGridy(); 
  can->SetGridx(); 
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
  htmlFile << "  <title>Detector Diagnostics LED Monitor</title> " << std::endl;
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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics LED Monitor</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  /////////////////////////////////////////// 
  htmlFile << "<table width=100% border=1>" << std::endl;
  htmlFile << "<tr>" << std::endl;
  htmlFile << "<td class=\"s0\" width=15% align=\"center\">SebDet</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Missing</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Unstable</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">low/no LED signal</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Bad Timing</td>" << std::endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Bad LED signal</td>" << std::endl;
  htmlFile << "</tr><tr>" << std::endl;
  int ind1=0,ind2=0,ind3=0,ind4=0,ind5=0;
  htmlFile << "<td class=\"s1\" align=\"center\">HB+</td>" << std::endl;
  ind1=3; if(newHBP[0]==0) ind1=2; if(newHBP[0]>0 && newHBP[0]<=12) ind1=1; if(newHBP[0]>=12 && HBP[0]<1296) ind1=0; 
  ind2=3; if(newHBP[1]==0) ind2=2; if(newHBP[1]>0)  ind2=1; if(newHBP[1]>21)  ind2=0; 
  ind3=3; if(newHBP[2]==0) ind3=2; if(newHBP[2]>0)  ind3=1; if(newHBP[2]>21)  ind3=0;
  ind4=3; if((newHBP[3]+newHBP[4])==0) ind4=2; if((newHBP[3]+newHBP[4])>0)  ind4=1; if((newHBP[3]+newHBP[4])>21)  ind4=0;
  ind5=3; if((newHBP[5]+newHBP[6])==0) ind5=2; if((newHBP[5]+newHBP[6])>0)  ind5=1; if((newHBP[5]+newHBP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;  
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HBP[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HBP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HBP[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HBP[5]+HBP[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HBP[3]+HBP[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HB-</td>" << std::endl;
  ind1=3; if(newHBM[0]==0) ind1=2; if(newHBM[0]>0 && newHBM[0]<=12) ind1=1; if(newHBM[0]>=12 && HBM[0]<1296) ind1=0; 
  ind2=3; if(newHBM[1]==0) ind2=2; if(newHBM[1]>0)  ind2=1; if(newHBM[1]>21)  ind2=0; 
  ind3=3; if(newHBM[2]==0) ind3=2; if(newHBM[2]>0)  ind3=1; if(newHBM[2]>21)  ind3=0;
  ind4=3; if((newHBM[3]+newHBM[4])==0) ind4=2; if((newHBM[3]+newHBM[4])>0)  ind4=1; if((newHBM[3]+newHBM[4])>21)  ind4=0;
  ind5=3; if((newHBM[5]+newHBM[6])==0) ind5=2; if((newHBM[5]+newHBM[6])>0)  ind5=1; if((newHBM[5]+newHBM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HBM[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HBM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HBM[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HBM[5]+HBM[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HBM[3]+HBM[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE+</td>" << std::endl;
  ind1=3; if(newHEP[0]==0) ind1=2; if(newHEP[0]>0 && newHEP[0]<=12) ind1=1; if(newHEP[0]>=12 && HEP[0]<1296) ind1=0; 
  ind2=3; if(newHEP[1]==0) ind2=2; if(newHEP[1]>0)  ind2=1; if(newHEP[1]>21)  ind2=0; 
  ind3=3; if(newHEP[2]==0) ind3=2; if(newHEP[2]>0)  ind3=1; if(newHEP[2]>21)  ind3=0;
  ind4=3; if((newHEP[3]+newHEP[4])==0) ind4=2; if((newHEP[3]+newHEP[4])>0)  ind4=1; if((newHEP[3]+newHEP[4])>21)  ind4=0;
  ind5=3; if((newHEP[5]+newHEP[6])==0) ind5=2; if((newHEP[5]+newHEP[6])>0)  ind5=1; if((newHEP[5]+newHEP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HEP[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HEP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HEP[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HEP[5]+HEP[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HEP[3]+HEP[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE-</td>" << std::endl;
  ind1=3; if(newHEM[0]==0) ind1=2; if(newHEM[0]>0 && newHEM[0]<=12) ind1=1; if(newHEM[0]>=12 && HEM[0]<1296) ind1=0; 
  ind2=3; if(newHEM[1]==0) ind2=2; if(newHEM[1]>0)  ind2=1; if(newHEM[1]>21)  ind2=0; 
  ind3=3; if(newHEM[2]==0) ind3=2; if(newHEM[2]>0)  ind3=1; if(newHEM[2]>21)  ind3=0;
  ind4=3; if((newHEM[3]+newHEM[4])==0) ind4=2; if((newHEM[3]+newHEM[4])>0)  ind4=1; if((newHEM[3]+newHEM[4])>21)  ind4=0;
  ind5=3; if((newHEM[5]+newHEM[6])==0) ind5=2; if((newHEM[5]+newHEM[6])>0)  ind5=1; if((newHEM[5]+newHEM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HEM[0] <<" (1296)</td>" << std::endl;
  htmlFile << state[ind2] << HEM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HEM[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HEM[5]+HEM[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HEM[3]+HEM[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF+</td>" << std::endl;
  ind1=3; if(newHFP[0]==0) ind1=2; if(newHFP[0]>0 && newHFP[0]<=12) ind1=1; if(newHFP[0]>=12 && HFP[0]<864) ind1=0; 
  ind2=3; if(newHFP[1]==0) ind2=2; if(newHFP[1]>0)  ind2=1; if(newHFP[1]>21)  ind2=0; 
  ind3=3; if(newHFP[2]==0) ind3=2; if(newHFP[2]>0)  ind3=1; if(newHFP[2]>21)  ind3=0;
  ind4=3; if((newHFP[3]+newHFP[4])==0) ind4=2; if((newHFP[3]+newHFP[4])>0)  ind4=1; if((newHFP[3]+newHFP[4])>21)  ind4=0;
  ind5=3; if((newHFP[5]+newHFP[6])==0) ind5=2; if((newHFP[5]+newHFP[6])>0)  ind5=1; if((newHFP[5]+newHFP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HFP[0] <<" (864)</td>" << std::endl;
  htmlFile << state[ind2] << HFP[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HFP[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HFP[5]+HFP[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HFP[3]+HFP[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF-</td>" << std::endl;
  ind1=3; if(newHFM[0]==0) ind1=2; if(newHFM[0]>0 && newHFM[0]<=12) ind1=1; if(newHFM[0]>=12 && HFM[0]<864) ind1=0; 
  ind2=3; if(newHFM[1]==0) ind2=2; if(newHFM[1]>0)  ind2=1; if(newHFM[1]>21)  ind2=0; 
  ind3=3; if(newHFM[2]==0) ind3=2; if(newHFM[2]>0)  ind3=1; if(newHFM[2]>21)  ind3=0;
  ind4=3; if((HFM[3]+HFM[4])==0) ind4=2; if((HFM[3]+HFM[4])>0)  ind4=1; if((HFM[3]+HFM[4])>21)  ind4=0;
  ind5=3; if((HFM[5]+HFM[6])==0) ind5=2; if((HFM[5]+HFM[6])>0)  ind5=1; if((HFM[5]+HFM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HFM[0] <<" (864)</td>" << std::endl;
  htmlFile << state[ind2] << HFM[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HFM[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HFM[5]+HFM[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HFM[3]+HFM[4] <<"</td>" << std::endl;
   
  htmlFile << "</tr><tr>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HO</td>" << std::endl;
  ind1=3; if(newHO[0]==0) ind1=2; if(newHO[0]>0 && newHO[0]<=12) ind1=1; if(newHO[0]>=12 && HO[0]<2160) ind1=0; 
  ind2=3; if(newHO[1]==0) ind2=2; if(newHO[1]>0)  ind2=1; if(newHO[1]>21)  ind2=0; 
  ind3=3; if(newHO[2]==0) ind3=2; if(newHO[2]>0)  ind3=1; if(newHO[2]>21)  ind3=0;
  ind4=3; if((newHO[3]+newHO[4])==0) ind4=2; if((newHO[3]+newHO[4])>0)  ind4=1; if((newHO[3]+newHO[4])>21)  ind4=0;
  ind5=3; if((newHO[5]+newHO[6])==0) ind5=2; if((newHO[5]+newHO[6])>0)  ind5=1; if((newHO[5]+newHO[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 

  htmlFile << state[ind1] << HO[0] <<" (2160)</td>" << std::endl;
  htmlFile << state[ind2] << HO[1] <<"</td>" << std::endl;
  htmlFile << state[ind3] << HO[2] <<"</td>" << std::endl;
  htmlFile << state[ind5] << HO[5]+HO[6] <<"</td>" << std::endl;
  htmlFile << state[ind4] << HO[3]+HO[4] <<"</td>" << std::endl;
  
  htmlFile << "</tr></table>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  /////////////////////////////////////////// 
  if((MissingCnt+UnstableCnt+UnstableLEDCnt+BadTimingCnt)>0){
      htmlFile << "<table width=100% border=1><tr>" << std::endl;
      if(MissingCnt>0)    htmlFile << "<td><a href=\"" << "Missing.html" <<"\">list of missing channels</a></td>";
      if(UnstableCnt>0)   htmlFile << "<td><a href=\"" << "Unstable.html" <<"\">list of unstable channels</a></td>";
      if(UnstableLEDCnt>0)htmlFile << "<td><a href=\"" << "UnstableLED.html" <<"\">list of low LED signal channels</a></td>";
      if(BadTimingCnt>0)htmlFile << "<td><a href=\"" << "BadTiming.html" <<"\">list of Bad Timing channels</a></td>";
      htmlFile << "</tr></table>" << std::endl;
  }
  ///////////////////////////////////////////   

  ///////////////////////////////////////////   
  htmlFile << "<h2 align=\"center\">Summary LED plots</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  Time2Dhbhehf->SetMaximum(6);
  Time2Dho->SetMaximum(6);
  Time2Dhbhehf->SetNdivisions(36,"Y");
  Time2Dho->SetNdivisions(36,"Y");
  Time2Dhbhehf->SetStats(0);
  Time2Dho->SetStats(0);
  Time2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "led_timing_hbhehf.gif").c_str());
  Time2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "led_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timing_hbhehf.gif\" alt=\"led timing distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timing_ho.gif\" alt=\"led timing distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  Energy2Dhbhehf->SetStats(0);
  Energy2Dho->SetStats(0);
  Energy2Dhbhehf->SetNdivisions(36,"Y");
  Energy2Dho->SetNdivisions(36,"Y");
  Energy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "led_energy_hbhehf.gif").c_str());
  Energy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "led_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energy_hbhehf.gif\" alt=\"led energy distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energy_ho.gif\" alt=\"led energy distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  can->SetGridy(false);
  htmlFile << "<tr align=\"left\">" << std::endl;
  Energy->Draw();    can->SaveAs((htmlDir + "led_energy_distribution.gif").c_str());
  EnergyRMS->Draw(); can->SaveAs((htmlDir + "led_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energy_distribution.gif\" alt=\"led energy distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energy_rms_distribution.gif\" alt=\"led energy rms distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  Timing->Draw();    can->SaveAs((htmlDir + "led_timing_distribution.gif").c_str());
  TimingRMS->Draw(); can->SaveAs((htmlDir + "led_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timing_distribution.gif\" alt=\"led timing distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timing_rms_distribution.gif\" alt=\"led timing rms distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  EnergyHF->Draw();    can->SaveAs((htmlDir + "led_energyhf_distribution.gif").c_str());
  EnergyRMSHF->Draw(); can->SaveAs((htmlDir + "led_energyhf_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energyhf_distribution.gif\" alt=\"hf led energy distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energyhf_rms_distribution.gif\" alt=\"hf led energy rms distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  TimingHF->Draw();    can->SaveAs((htmlDir + "led_timinghf_distribution.gif").c_str());
  TimingRMSHF->Draw(); can->SaveAs((htmlDir + "led_timinghf_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timinghf_distribution.gif\" alt=\"hf led timing distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timinghf_rms_distribution.gif\" alt=\"hf led timing rms distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  
  ///////////////////////////////////////////  
  htmlFile << "<h2 align=\"center\">Stability LED plots (Reference run "<<ref_run<<")</h2>" << std::endl;
  htmlFile << "<table width=100% border=0><tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  HBPphi->GetXaxis()->SetNdivisions(418,kFALSE);
  HBMphi->GetXaxis()->SetNdivisions(418,kFALSE);
  HEPphi->GetXaxis()->SetNdivisions(418,kFALSE);
  HEMphi->GetXaxis()->SetNdivisions(418,kFALSE);
  
  HBPphi->SetMarkerColor(kRed);
  HBPphi->SetMarkerStyle(23);
  HBPphi->SetXTitle("HPD Index = RBX*4+RM");
  HBMphi->SetMarkerColor(kRed);
  HBMphi->SetMarkerStyle(23);
  HBMphi->SetXTitle("HPD Index = RBX*4+RM");
  HBPphi->Draw();    can->SaveAs((htmlDir + "led_hbp_distribution.gif").c_str());
  HBMphi->Draw();    can->SaveAs((htmlDir + "led_hbm_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_hbp_distribution.gif\" alt=\"led hbp distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hbm_distribution.gif\" alt=\"led hbm distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  HEPphi->SetMarkerColor(kRed);
  HEPphi->SetMarkerStyle(23);
  HEPphi->SetXTitle("HPD Index = RBX*4+RM");
  HEMphi->SetMarkerColor(kRed);
  HEMphi->SetMarkerStyle(23);
  HEMphi->SetXTitle("HPD Index = RBX*4+RM");
  HEPphi->Draw();    can->SaveAs((htmlDir + "led_hep_distribution.gif").c_str());
  HEMphi->Draw();    can->SaveAs((htmlDir + "led_hem_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_hep_distribution.gif\" alt=\"led hep distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hem_distribution.gif\" alt=\"led hem distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  HFPphi->SetMarkerColor(kRed);
  HFPphi->SetMarkerStyle(23);
  HFPphi->SetXTitle("RM Index = RoBox*3+RM");
  HFMphi->SetMarkerColor(kRed);
  HFMphi->SetMarkerStyle(23);
  HFPphi->GetXaxis()->SetNdivisions(312,kFALSE);
  HFMphi->GetXaxis()->SetNdivisions(312,kFALSE);

  HFMphi->SetXTitle("RM Index = RoBox*3+RM");
  HFPphi->Draw();    can->SaveAs((htmlDir + "led_hfp_distribution.gif").c_str());
  HFMphi->Draw();    can->SaveAs((htmlDir + "led_hfm_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_hfp_distribution.gif\" alt=\"led hfp distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hfm_distribution.gif\" alt=\"led hfm distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl; 

  htmlFile << "<tr align=\"left\">" << std::endl;
  HO1Pphi->SetMarkerColor(kRed);
  HO1Pphi->SetMarkerStyle(23);
  HO1Pphi->SetXTitle("HPD Index = RBX*4+RM");
  HO1Mphi->SetMarkerColor(kRed);
  HO1Mphi->SetMarkerStyle(23);
  HO1Mphi->GetXaxis()->SetNdivisions(412,kFALSE);
  HO1Pphi->GetXaxis()->SetNdivisions(412,kFALSE);

  HO1Mphi->SetXTitle("HPD Index = RBX*4+RM");
  HO1Pphi->Draw();    can->SaveAs((htmlDir + "led_ho1p_distribution.gif").c_str());
  HO1Mphi->Draw();    can->SaveAs((htmlDir + "led_ho1m_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_ho1p_distribution.gif\" alt=\"led ho1p distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_ho1m_distribution.gif\" alt=\"led ho1m distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
   
  htmlFile << "<tr align=\"left\">" << std::endl;
  HO2Pphi->SetMarkerColor(kRed);
  HO2Pphi->SetMarkerStyle(23);
  HO2Pphi->SetXTitle("HPD Index = RBX*4+RM");
  HO2Mphi->SetMarkerColor(kRed);
  HO2Mphi->SetMarkerStyle(23);
  HO2Mphi->GetXaxis()->SetNdivisions(412,kFALSE);
  HO2Pphi->GetXaxis()->SetNdivisions(412,kFALSE);

  HO2Mphi->SetXTitle("HPD Index = RBX*4+RM");
  HO2Pphi->Draw();    can->SaveAs((htmlDir + "led_ho2p_distribution.gif").c_str());
  HO2Mphi->Draw();    can->SaveAs((htmlDir + "led_ho2m_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_ho2p_distribution.gif\" alt=\"led ho2p distribution\">   </td>" << std::endl;
  htmlFile << "<td align=\"center\"><img src=\"led_ho2m_distribution.gif\" alt=\"led ho2m distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  HO0phi->SetMarkerColor(kRed);
  HO0phi->SetMarkerStyle(23);
  HO0phi->SetXTitle("HPD Index = RBX*4+RM");
  HO0phi->GetXaxis()->SetNdivisions(412,kFALSE);
  HO0phi->Draw();    can->SaveAs((htmlDir + "led_ho0_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_ho0_distribution.gif\" alt=\"led ho0 distribution\">   </td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "</table>" << std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();
  can->Close();
}

HcalDetDiagLEDClient::~HcalDetDiagLEDClient()
{}
