#include "DQM/HcalMonitorClient/interface/HcalDetDiagLEDClient.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include <TPaveStats.h>

HcalDetDiagLEDClient::HcalDetDiagLEDClient(){}
HcalDetDiagLEDClient::~HcalDetDiagLEDClient(){}
void HcalDetDiagLEDClient::beginJob(){}
void HcalDetDiagLEDClient::beginRun(){}
void HcalDetDiagLEDClient::endJob(){} 
void HcalDetDiagLEDClient::endRun(){} 
void HcalDetDiagLEDClient::cleanup(){} 
void HcalDetDiagLEDClient::analyze(){} 
void HcalDetDiagLEDClient::report(){} 
void HcalDetDiagLEDClient::resetAllME(){} 
void HcalDetDiagLEDClient::createTests(){}
void HcalDetDiagLEDClient::loadHistograms(TFile* infile){}

void HcalDetDiagLEDClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  HcalBaseClient::init(ps,dbe,clientName);
  status=0;
} 

void HcalDetDiagLEDClient::getHistograms(){
  std::string folder="HcalDetDiagLEDMonitor/Summary Plots/";
  Energy        =getHisto(folder+"HBHEHO LED Energy Distribution",              process_, dbe_, debug_,cloneME_);
  Timing        =getHisto(folder+"HBHEHO LED Timing Distribution",              process_, dbe_, debug_,cloneME_);
  EnergyRMS     =getHisto(folder+"HBHEHO LED Energy RMS/Energy Distribution",   process_, dbe_, debug_,cloneME_);
  TimingRMS     =getHisto(folder+"HBHEHO LED Timing RMS Distribution",          process_, dbe_, debug_,cloneME_);
  EnergyHF      =getHisto(folder+"HF LED Energy Distribution",              process_, dbe_, debug_,cloneME_);
  TimingHF      =getHisto(folder+"HF LED Timing Distribution",              process_, dbe_, debug_,cloneME_);
  EnergyRMSHF   =getHisto(folder+"HF LED Energy RMS/Energy Distribution",   process_, dbe_, debug_,cloneME_);
  TimingRMSHF   =getHisto(folder+"HF LED Timing RMS Distribution",          process_, dbe_, debug_,cloneME_);
  EnergyCorr    =getHisto(folder+"LED Energy Corr(PinDiod) Distribution",process_, dbe_, debug_,cloneME_);
  Time2Dhbhehf  =getHisto2(folder+"LED Timing HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Time2Dho      =getHisto2(folder+"LED Timing HO",                       process_, dbe_, debug_,cloneME_);
  Energy2Dhbhehf=getHisto2(folder+"LED Energy HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Energy2Dho    =getHisto2(folder+"LED Energy HO",                       process_, dbe_, debug_,cloneME_);
  HBPphi        =getHisto2(folder+"HBP Average over HPD LED Ref",        process_, dbe_, debug_,cloneME_);
  HBMphi        =getHisto2(folder+"HBM Average over HPD LED Ref",        process_, dbe_, debug_,cloneME_);
  HEPphi        =getHisto2(folder+"HEP Average over HPD LED Ref",        process_, dbe_, debug_,cloneME_);
  HEMphi        =getHisto2(folder+"HEM Average over HPD LED Ref",        process_, dbe_, debug_,cloneME_);
  HFPphi        =getHisto2(folder+"HFP Average over RM LED Ref",      process_, dbe_, debug_,cloneME_);
  HFMphi        =getHisto2(folder+"HFM Average over RM LED Ref",      process_, dbe_, debug_,cloneME_);
  HO0phi        =getHisto2(folder+"HO0 Average over HPD LED Ref",        process_, dbe_, debug_,cloneME_);
  HO1Pphi       =getHisto2(folder+"HO1P Average over HPD LED Ref",       process_, dbe_, debug_,cloneME_);
  HO2Pphi       =getHisto2(folder+"HO2P Average over HPD LED Ref",       process_, dbe_, debug_,cloneME_);
  HO1Mphi       =getHisto2(folder+"HO1M Average over HPD LED Ref",       process_, dbe_, debug_,cloneME_);
  HO2Mphi       =getHisto2(folder+"HO2M Average over HPD LED Ref",       process_, dbe_, debug_,cloneME_);
 
  getSJ6histos("HcalDetDiagLEDMonitor/Summary Plots/","Channel LED Energy", ChannelsLEDEnergy);
  getSJ6histos("HcalDetDiagLEDMonitor/Summary Plots/","Channel LED Energy Reference", ChannelsLEDEnergyRef);
    
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status Missing Channels", ChannelStatusMissingChannels);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status Unstable Channels",ChannelStatusUnstableChannels);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status Unstable LED",     ChannelStatusUnstableLEDsignal);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status LED Mean",         ChannelStatusLEDMean);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status LED RMS",          ChannelStatusLEDRMS);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status Time Mean",        ChannelStatusTimeMean);
  getSJ6histos("HcalDetDiagLEDMonitor/channel status/","Channel Status Time RMS",         ChannelStatusTimeRMS);
 
  MonitorElement* me = dbe_->get("Hcal/HcalDetDiagLEDMonitor/HcalDetDiagLEDMonitor Event Number");
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  me = dbe_->get("Hcal/HcalDetDiagLEDMonitor/HcalDetDiagLEDMonitor Reference Run");
  if(me) {
    string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
  }
} 
bool HcalDetDiagLEDClient::haveOutput(){
    getHistograms();
    if(ievt_>100) return true;
    return false; 
}
int  HcalDetDiagLEDClient::SummaryStatus(){
    return status;
}
double HcalDetDiagLEDClient::get_channel_status(char *subdet,int eta,int phi,int depth,int type){
   int ind=-1;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return -1.0;
   if(type==1) return ChannelStatusMissingChannels[ind]  ->GetBinContent(eta+42,phi+1);
   if(type==2) return ChannelStatusUnstableChannels[ind] ->GetBinContent(eta+42,phi+1);
   if(type==3) return ChannelStatusUnstableLEDsignal[ind]->GetBinContent(eta+42,phi+1);
   if(type==4) return ChannelStatusLEDMean[ind]          ->GetBinContent(eta+42,phi+1);
   if(type==5) return ChannelStatusLEDRMS[ind]           ->GetBinContent(eta+42,phi+1);
   if(type==6) return ChannelStatusTimeMean[ind]         ->GetBinContent(eta+42,phi+1);
   if(type==7) return ChannelStatusTimeRMS[ind]          ->GetBinContent(eta+42,phi+1);
   return -1.0;
}
double HcalDetDiagLEDClient::get_energy(char *subdet,int eta,int phi,int depth,int type){
   int ind=-1;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return -1.0;
   if(type==1) return ChannelsLEDEnergy[ind]  ->GetBinContent(eta+42,phi+1);
   if(type==2) return ChannelsLEDEnergyRef[ind] ->GetBinContent(eta+42,phi+1);
   return -1.0;
}

static void printTableHeader(ofstream& file,char * header){
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
static void printTableLine(ofstream& file,int ind,HcalDetId& detid,HcalFrontEndId& lmap_entry,HcalElectronicsId &emap_entry,char *comment=""){
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
   char *raw_class;
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
void HcalDetDiagLEDClient::htmlOutput(int runNo, string htmlDir, string htmlName){
int  MissingCnt=0;
int  UnstableCnt=0;
int  UnstableLEDCnt=0;
int  BadTimingCnt=0;
int  BadCnt=0; 
int  HBP[7]={0,0,0,0,0,0,0}; 
int  HBM[7]={0,0,0,0,0,0,0};  
int  HEP[7]={0,0,0,0,0,0,0};  
int  HEM[7]={0,0,0,0,0,0,0};  
int  HFP[7]={0,0,0,0,0,0,0}; 
int  HFM[7]={0,0,0,0,0,0,0}; 
int  HO[7] ={0,0,0,0,0,0,0};  
char *subdet[4]={"HB","HE","HO","HF"};

  HcalLogicalMapGenerator gen;
  HcalLogicalMap lmap(gen.createMap());
  HcalElectronicsMap emap=lmap.generateHcalElectronicsMap();
  
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
	      if(sd==0)  if(eta>0) HBP[i]++; else HBM[i]++; 
	      if(sd==1)  if(eta>0) HEP[i]++; else HEM[i]++; 
	      if(sd==2)  HO[i]++; 
	      if(sd==3)  if(eta>0) HFP[i]++; else HFM[i]++; 
           }
        }
     }
  }
 
  // missing channels list
  ofstream Missing;
  Missing.open((htmlDir + "Missing_"+htmlName).c_str());
  printTableHeader(Missing,"Missing Channels list");
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && ((HBM[0]+HBP[0])==0 || (HBM[0]+HBP[0])==(1296*2))) continue;
      if(sd==1 && ((HEM[0]+HEP[0])==0 || (HEM[0]+HEP[0])==(1296*2))) continue;
      if(sd==2 && ((HO[0])==0 || HO[0]==2160))                      continue;
      if(sd==3 && ((HFM[0]+HFP[0])==0 || (HFM[0]+HFP[0])==(864*2))) continue;
      Missing << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2; if(HBM[0]==1296) feta=0; if(HBP[0]==1296) teta=0;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3; if(HEM[0]==1296) feta=0; if(HEP[0]==1296) teta=0;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4; if(HO[0] ==2160) {feta=0; teta=0; }} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2; if(HFM[0]==864)  feta=0; if(HFP[0]==864)  teta=0; } 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double missing =get_channel_status(subdet[sd],eta,phi,depth,1);
         if(missing>0){
            try{
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(Missing,cnt++,*detid,lmap_entry,emap_entry); MissingCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(Missing);
  Missing.close();

  // Bad timing channels list
  ofstream BadTiming;
  BadTiming.open((htmlDir + "BadTiming_"+htmlName).c_str());
  printTableHeader(BadTiming,"Bad Timing Channels list");
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[5]+HBP[5])==0) continue;
      if(sd==1 && (HEM[5]+HEP[5])==0) continue;
      if(sd==2 && (HO[5])==0)         continue;
      if(sd==3 && (HFM[5]+HFP[5])==0) continue;
      BadTiming << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << endl;
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
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(BadTiming,cnt++,*detid,lmap_entry,emap_entry,comment); BadTimingCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(BadTiming);
  BadTiming.close();
  
  // unstable channels list
  ofstream Unstable;
  Unstable.open((htmlDir + "Unstable_"+htmlName).c_str());
  printTableHeader(Unstable,"Low LED signal Channels list");
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[1]+HBP[1])==0) continue;
      if(sd==1 && (HEM[1]+HEP[1])==0) continue;
      if(sd==2 && (HO[1])==0)         continue;
      if(sd==3 && (HFM[1]+HFP[1])==0) continue;
      Unstable << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << endl;
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
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(Unstable,cnt++,*detid,lmap_entry,emap_entry,comment); UnstableCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(Unstable);
  Unstable.close();
  
  // unstable LED signal list
  ofstream BadLED;
  BadLED.open((htmlDir + "UnstableLED_"+htmlName).c_str());
  printTableHeader(BadLED,"Unstable LED signal channels list");
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[2]+HBP[2])==0) continue;
      if(sd==1 && (HEM[2]+HEP[2])==0) continue;
      if(sd==2 &&  (HO[2])==0)        continue;
      if(sd==3 && (HFM[2]+HFP[2])==0) continue;
      BadLED << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << endl;
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
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(BadLED,cnt++,*detid,lmap_entry,emap_entry,comment); UnstableLEDCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(BadLED);
  BadLED.close();
  
  if (debug_>0) cout << "<HcalDetDiagLEDClient::htmlOutput> Preparing  html output ..." << endl;
  if(!dbe_) return;
  string client = "HcalDetDiagLEDClient";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
 
  TCanvas *can=new TCanvas("HcalDetDiagLEDClient","HcalDetDiagLEDClient",0,0,500,350);
  can->cd();
  can->SetGridy();
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());
  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Detector Diagnostics LED Monitor</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<style type=\"text/css\">"<< endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FF7700; text-align: center;}"<< endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: red; }"<< endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: yellow; }"<< endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: green; }"<< endl;
  htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: silver; }"<< endl;
  char *state[4]={"<td class=\"s2\" align=\"center\">",
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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics LED Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  /////////////////////////////////////////// 
  htmlFile << "<table width=100% border=1>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td class=\"s0\" width=15% align=\"center\">SebDet</td>" << endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Missing</td>" << endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Unstable</td>" << endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">low/no LED signal</td>" << endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Bad Timing</td>" << endl;
  htmlFile << "<td class=\"s0\" width=17% align=\"center\">Bad LED signal</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  int ind1=0,ind2=0,ind3=0,ind4=0,ind5=0;
  htmlFile << "<td class=\"s1\" align=\"center\">HB+</td>" << endl;
  ind1=3; if(HBP[0]==0) ind1=2; if(HBP[0]>0 && HBP[0]<=12) ind1=1; if(HBP[0]>=12 && HBP[0]<1296) ind1=0; 
  ind2=3; if(HBP[1]==0) ind2=2; if(HBP[1]>0)  ind2=1; if(HBP[1]>21)  ind2=0; 
  ind3=3; if(HBP[2]==0) ind3=2; if(HBP[2]>0)  ind3=1; if(HBP[2]>21)  ind3=0;
  ind4=3; if((HBP[3]+HBP[4])==0) ind4=2; if((HBP[3]+HBP[4])>0)  ind4=1; if((HBP[3]+HBP[4])>21)  ind4=0;
  ind5=3; if((HBP[5]+HBP[6])==0) ind5=2; if((HBP[5]+HBP[6])>0)  ind5=1; if((HBP[5]+HBP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;  
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status=1; 
  htmlFile << state[ind1] << HBP[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HBP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HBP[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HBP[5]+HBP[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HBP[3]+HBP[4] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HB-</td>" << endl;
  ind1=3; if(HBM[0]==0) ind1=2; if(HBM[0]>0 && HBM[0]<=12) ind1=1; if(HBM[0]>=12 && HBM[0]<1296) ind1=0; 
  ind2=3; if(HBM[1]==0) ind2=2; if(HBM[1]>0)  ind2=1; if(HBM[1]>21)  ind2=0; 
  ind3=3; if(HBM[2]==0) ind3=2; if(HBM[2]>0)  ind3=1; if(HBM[2]>21)  ind3=0;
  ind4=3; if((HBM[3]+HBM[4])==0) ind4=2; if((HBM[3]+HBM[4])>0)  ind4=1; if((HBM[3]+HBM[4])>21)  ind4=0;
  ind5=3; if((HBM[5]+HBM[6])==0) ind5=2; if((HBM[5]+HBM[6])>0)  ind5=1; if((HBM[5]+HBM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HBM[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HBM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HBM[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HBM[5]+HBM[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HBM[3]+HBM[4] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE+</td>" << endl;
  ind1=3; if(HEP[0]==0) ind1=2; if(HEP[0]>0 && HEP[0]<=12) ind1=1; if(HEP[0]>=12 && HEP[0]<1296) ind1=0; 
  ind2=3; if(HEP[1]==0) ind2=2; if(HEP[1]>0)  ind2=1; if(HEP[1]>21)  ind2=0; 
  ind3=3; if(HEP[2]==0) ind3=2; if(HEP[2]>0)  ind3=1; if(HEP[2]>21)  ind3=0;
  ind4=3; if((HEP[3]+HEP[4])==0) ind4=2; if((HEP[3]+HEP[4])>0)  ind4=1; if((HEP[3]+HEP[4])>21)  ind4=0;
  ind5=3; if((HEP[5]+HEP[6])==0) ind5=2; if((HEP[5]+HEP[6])>0)  ind5=1; if((HEP[5]+HEP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HEP[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HEP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HEP[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HEP[5]+HEP[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HEP[3]+HEP[4] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE-</td>" << endl;
  ind1=3; if(HEM[0]==0) ind1=2; if(HEM[0]>0 && HEM[0]<=12) ind1=1; if(HEM[0]>=12 && HEM[0]<1296) ind1=0; 
  ind2=3; if(HEM[1]==0) ind2=2; if(HEM[1]>0)  ind2=1; if(HEM[1]>21)  ind2=0; 
  ind3=3; if(HEM[2]==0) ind3=2; if(HEM[2]>0)  ind3=1; if(HEM[2]>21)  ind3=0;
  ind4=3; if((HEM[3]+HEM[4])==0) ind4=2; if((HEM[3]+HEM[4])>0)  ind4=1; if((HEM[3]+HEM[4])>21)  ind4=0;
  ind5=3; if((HEM[5]+HEM[6])==0) ind5=2; if((HEM[5]+HEM[6])>0)  ind5=1; if((HEM[5]+HEM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HEM[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HEM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HEM[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HEM[5]+HEM[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HEM[3]+HEM[4] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF+</td>" << endl;
  ind1=3; if(HFP[0]==0) ind1=2; if(HFP[0]>0 && HFP[0]<=12) ind1=1; if(HFP[0]>=12 && HFP[0]<864) ind1=0; 
  ind2=3; if(HFP[1]==0) ind2=2; if(HFP[1]>0)  ind2=1; if(HFP[1]>21)  ind2=0; 
  ind3=3; if(HFP[2]==0) ind3=2; if(HFP[2]>0)  ind3=1; if(HFP[2]>21)  ind3=0;
  ind4=3; if((HFP[3]+HFP[4])==0) ind4=2; if((HFP[3]+HFP[4])>0)  ind4=1; if((HFP[3]+HFP[4])>21)  ind4=0;
  ind5=3; if((HFP[5]+HFP[6])==0) ind5=2; if((HFP[5]+HFP[6])>0)  ind5=1; if((HFP[5]+HFP[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HFP[0] <<" (864)</td>" << endl;
  htmlFile << state[ind2] << HFP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HFP[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HFP[5]+HFP[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HFP[3]+HFP[4] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF-</td>" << endl;
  ind1=3; if(HFM[0]==0) ind1=2; if(HFM[0]>0 && HFM[0]<=12) ind1=1; if(HFM[0]>=12 && HFM[0]<864) ind1=0; 
  ind2=3; if(HFM[1]==0) ind2=2; if(HFM[1]>0)  ind2=1; if(HFM[1]>21)  ind2=0; 
  ind3=3; if(HFM[2]==0) ind3=2; if(HFM[2]>0)  ind3=1; if(HFM[2]>21)  ind3=0;
  ind4=3; if((HFM[3]+HFM[4])==0) ind4=2; if((HFM[3]+HFM[4])>0)  ind4=1; if((HFM[3]+HFM[4])>21)  ind4=0;
  ind5=3; if((HFM[5]+HFM[6])==0) ind5=2; if((HFM[5]+HFM[6])>0)  ind5=1; if((HFM[5]+HFM[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HFM[0] <<" (864)</td>" << endl;
  htmlFile << state[ind2] << HFM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HFM[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HFM[5]+HFM[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HFM[3]+HFM[4] <<"</td>" << endl;
   
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HO</td>" << endl;
  ind1=3; if(HO[0]==0) ind1=2; if(HO[0]>0 && HO[0]<=12) ind1=1; if(HO[0]>=12 && HO[0]<2160) ind1=0; 
  ind2=3; if(HO[1]==0) ind2=2; if(HO[1]>0)  ind2=1; if(HO[1]>21)  ind2=0; 
  ind3=3; if(HO[2]==0) ind3=2; if(HO[2]>0)  ind3=1; if(HO[2]>21)  ind3=0;
  ind4=3; if((HO[3]+HO[4])==0) ind4=2; if((HO[3]+HO[4])>0)  ind4=1; if((HO[3]+HO[4])>21)  ind4=0;
  ind5=3; if((HO[5]+HO[6])==0) ind5=2; if((HO[5]+HO[6])>0)  ind5=1; if((HO[5]+HO[6])>21)  ind5=0;
  if(ind1==3) ind2=ind3=ind4=ind5=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status=1; 
  htmlFile << state[ind1] << HO[0] <<" (2160)</td>" << endl;
  htmlFile << state[ind2] << HO[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HO[2] <<"</td>" << endl;
  htmlFile << state[ind5] << HO[5]+HO[6] <<"</td>" << endl;
  htmlFile << state[ind4] << HO[3]+HO[4] <<"</td>" << endl;
  
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  /////////////////////////////////////////// 
  if((MissingCnt+UnstableCnt+BadCnt)>0){
      htmlFile << "<table width=100% border=1><tr>" << endl;
      if(MissingCnt>0)    htmlFile << "<td><a href=\"" << ("Missing_"+htmlName).c_str() <<"\">list of missing channels</a></td>";
      if(UnstableCnt>0)   htmlFile << "<td><a href=\"" << ("Unstable_"+htmlName).c_str() <<"\">list of unstable channels</a></td>";
      if(UnstableLEDCnt>0)htmlFile << "<td><a href=\"" << ("UnstableLED_"+htmlName).c_str() <<"\">list of low LED signal channels</a></td>";
      if(BadTimingCnt>0)htmlFile << "<td><a href=\"" << ("BadTiming_"+htmlName).c_str() <<"\">list of Bad Timing channels</a></td>";
      htmlFile << "</tr></table>" << endl;
  }
  ///////////////////////////////////////////   
  htmlFile << "<h2 align=\"center\">Summary LED plots</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Time2Dhbhehf->SetMaximum(6);
  Time2Dho->SetMaximum(6);
  Time2Dhbhehf->SetNdivisions(36,"Y");
  Time2Dho->SetNdivisions(36,"Y");
  Time2Dhbhehf->SetStats(0);
  Time2Dho->SetStats(0);
  Time2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "led_timing_hbhehf.gif").c_str());
  Time2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "led_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timing_hbhehf.gif\" alt=\"led timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timing_ho.gif\" alt=\"led timing distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  Energy2Dhbhehf->SetStats(0);
  Energy2Dho->SetStats(0);
  Energy2Dhbhehf->SetNdivisions(36,"Y");
  Energy2Dho->SetNdivisions(36,"Y");
  Energy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "led_energy_hbhehf.gif").c_str());
  Energy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "led_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energy_hbhehf.gif\" alt=\"led energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energy_ho.gif\" alt=\"led energy distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  can->SetGridy(false);
  htmlFile << "<tr align=\"left\">" << endl;  
  Energy->Draw();    can->SaveAs((htmlDir + "led_energy_distribution.gif").c_str());
  EnergyRMS->Draw(); can->SaveAs((htmlDir + "led_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energy_distribution.gif\" alt=\"led energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energy_rms_distribution.gif\" alt=\"led energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Timing->Draw();    can->SaveAs((htmlDir + "led_timing_distribution.gif").c_str());
  TimingRMS->Draw(); can->SaveAs((htmlDir + "led_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timing_distribution.gif\" alt=\"led timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timing_rms_distribution.gif\" alt=\"led timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;  
  EnergyHF->Draw();    can->SaveAs((htmlDir + "led_energyhf_distribution.gif").c_str());
  EnergyRMSHF->Draw(); can->SaveAs((htmlDir + "led_energyhf_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_energyhf_distribution.gif\" alt=\"hf led energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_energyhf_rms_distribution.gif\" alt=\"hf led energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  TimingHF->Draw();    can->SaveAs((htmlDir + "led_timinghf_distribution.gif").c_str());
  TimingRMSHF->Draw(); can->SaveAs((htmlDir + "led_timinghf_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_timinghf_distribution.gif\" alt=\"hf led timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_timinghf_rms_distribution.gif\" alt=\"hf led timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  
  ///////////////////////////////////////////  
  can->SetGridy(); 
  can->SetGridx(); 
  htmlFile << "<h2 align=\"center\">Stability LED plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
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
  htmlFile << "<td align=\"center\"><img src=\"led_hbp_distribution.gif\" alt=\"led hbp distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hbm_distribution.gif\" alt=\"led hbm distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  
  
  htmlFile << "<tr align=\"left\">" << endl;
  HEPphi->SetMarkerColor(kRed);
  HEPphi->SetMarkerStyle(23);
  HEPphi->SetXTitle("HPD Index = RBX*4+RM");
  HEMphi->SetMarkerColor(kRed);
  HEMphi->SetMarkerStyle(23);
  HEMphi->SetXTitle("HPD Index = RBX*4+RM");
  HEPphi->Draw();    can->SaveAs((htmlDir + "led_hep_distribution.gif").c_str());
  HEMphi->Draw();    can->SaveAs((htmlDir + "led_hem_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_hep_distribution.gif\" alt=\"led hep distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hem_distribution.gif\" alt=\"led hem distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  
  
  htmlFile << "<tr align=\"left\">" << endl;
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
  htmlFile << "<td align=\"center\"><img src=\"led_hfp_distribution.gif\" alt=\"led hfp distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_hfm_distribution.gif\" alt=\"led hfm distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  

  htmlFile << "<tr align=\"left\">" << endl;
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
  htmlFile << "<td align=\"center\"><img src=\"led_ho1p_distribution.gif\" alt=\"led ho1p distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_ho1m_distribution.gif\" alt=\"led ho1m distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  
   
  htmlFile << "<tr align=\"left\">" << endl;
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
  htmlFile << "<td align=\"center\"><img src=\"led_ho2p_distribution.gif\" alt=\"led ho2p distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"led_ho2m_distribution.gif\" alt=\"led ho2m distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  
  
  htmlFile << "<tr align=\"left\">" << endl;
  HO0phi->SetMarkerColor(kRed);
  HO0phi->SetMarkerStyle(23);
  HO0phi->SetXTitle("HPD Index = RBX*4+RM");
  HO0phi->GetXaxis()->SetNdivisions(412,kFALSE);
  HO0phi->Draw();    can->SaveAs((htmlDir + "led_ho0_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"led_ho0_distribution.gif\" alt=\"led ho0 distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;  
  
  htmlFile << "</table>" << endl;
  
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();
} 

