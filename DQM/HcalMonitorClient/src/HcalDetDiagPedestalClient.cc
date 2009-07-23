#include "DQM/HcalMonitorClient/interface/HcalDetDiagPedestalClient.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include <TPaveStats.h>

static int IsKnownBadChannel(std::string subdet,int eta,int phi,int depth){
  if(subdet.compare("HO")==0 && eta==5    && phi==35 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==6    && phi==35 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==5    && phi==36 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==6    && phi==36 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-4   && phi==37 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==5    && phi==37 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==6    && phi==37 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==38 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-9   && phi==38 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-4   && phi==38 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==39 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-9   && phi==39 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==40 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-9   && phi==40 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==41 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==42 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-10  && phi==43 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-3   && phi==59 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-2   && phi==59 && depth==4) return 1;
  if(subdet.compare("HO")==0 && eta==-1   && phi==59 && depth==4) return 1;
  
  if(subdet.compare("HO")==0 && eta==1   && phi==4 && depth==4) return 3;
  if(subdet.compare("HO")==0 && eta==3   && phi==4 && depth==4) return 3;
  if(subdet.compare("HO")==0 && eta==15  && phi==24 && depth==4) return 3;
  
  if(subdet.compare("HF")==0 && eta==41  && phi==47 && depth==2) return 3;
  if(subdet.compare("HF")==0 && eta==29  && phi==71 && depth==1) return 3;
  return 0;
}

HcalDetDiagPedestalClient::HcalDetDiagPedestalClient(){}
HcalDetDiagPedestalClient::~HcalDetDiagPedestalClient(){}
void HcalDetDiagPedestalClient::beginJob(){}
void HcalDetDiagPedestalClient::beginRun(){}
void HcalDetDiagPedestalClient::endJob(){} 
void HcalDetDiagPedestalClient::endRun(){} 
void HcalDetDiagPedestalClient::cleanup(){} 
void HcalDetDiagPedestalClient::analyze(){} 
void HcalDetDiagPedestalClient::report(){} 
void HcalDetDiagPedestalClient::resetAllME(){} 
void HcalDetDiagPedestalClient::createTests(){}
void HcalDetDiagPedestalClient::loadHistograms(TFile* infile){}

void HcalDetDiagPedestalClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  HcalBaseClient::init(ps,dbe,clientName);
  status=0;
} 

void HcalDetDiagPedestalClient::getHistograms(){
  std::string folder="HcalDetDiagPedestalMonitor/Summary Plots/";
  
  Pedestals2DRmsHBHEHF  =getHisto2(folder+"HBHEHF pedestal rms map",      process_, dbe_, debug_,cloneME_);
  Pedestals2DRmsHO      =getHisto2(folder+"HO pedestal rms map",          process_, dbe_, debug_,cloneME_);
  Pedestals2DHBHEHF     =getHisto2(folder+"HBHEHF pedestal mean map",     process_, dbe_, debug_,cloneME_);
  Pedestals2DHO         =getHisto2(folder+"HO pedestal mean map",         process_, dbe_, debug_,cloneME_);
  Pedestals2DErrorHBHEHF=getHisto2(folder+"HBHEHF pedestal problems map", process_, dbe_, debug_,cloneME_);
  Pedestals2DErrorHO    =getHisto2(folder+"HO pedestal problems map",     process_, dbe_, debug_,cloneME_);

  PedestalsAve4HB  =new TH1F(*getHisto(folder+"HB Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsAve4HE  =new TH1F(*getHisto(folder+"HE Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsAve4HO  =new TH1F(*getHisto(folder+"HO Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsAve4HF  =new TH1F(*getHisto(folder+"HF Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsAve4Simp=new TH1F(*getHisto(folder+"SIPM Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsAve4ZDC =new TH1F(*getHisto(folder+"ZDC Pedestal Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  
  PedestalsRefAve4HB  =new TH1F(*getHisto(folder+"HB Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsRefAve4HE  =new TH1F(*getHisto(folder+"HE Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsRefAve4HO  =new TH1F(*getHisto(folder+"HO Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsRefAve4HF  =new TH1F(*getHisto(folder+"HF Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsRefAve4Simp=new TH1F(*getHisto(folder+"SIPM Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  PedestalsRefAve4ZDC =new TH1F(*getHisto(folder+"ZDC Pedestal Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_));
  
  PedestalsRefAve4HB->SetLineColor(kGreen);
  PedestalsRefAve4HE->SetLineColor(kGreen);
  PedestalsRefAve4HO->SetLineColor(kGreen);
  PedestalsRefAve4HF->SetLineColor(kGreen);
  PedestalsRefAve4Simp->SetLineColor(kGreen);
  PedestalsRefAve4ZDC->SetLineColor(kGreen);
  PedestalsRefAve4HB->SetLineWidth(3);
  PedestalsRefAve4HE->SetLineWidth(3);
  PedestalsRefAve4HO->SetLineWidth(3);
  PedestalsRefAve4HF->SetLineWidth(3);
  PedestalsRefAve4Simp->SetLineWidth(3);
  PedestalsRefAve4ZDC->SetLineWidth(3);
  
  PedestalsAve4HBref=getHisto(folder+"HB Pedestal-Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_);
  PedestalsAve4HEref=getHisto(folder+"HE Pedestal-Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_);
  PedestalsAve4HOref=getHisto(folder+"HO Pedestal-Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_);
  PedestalsAve4HFref=getHisto(folder+"HF Pedestal-Reference Distribution (avarage over 4 caps)", process_, dbe_, debug_,cloneME_);
  
  PedestalsRmsHB=getHisto(folder+"HB Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHE=getHisto(folder+"HE Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHO=getHisto(folder+"HO Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHF=getHisto(folder+"HF Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsSimp=getHisto(folder+"SIPM Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsZDC =getHisto(folder+"ZDC Pedestal RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  
  PedestalsRmsRefHB=getHisto(folder+"HB Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefHE=getHisto(folder+"HE Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefHO=getHisto(folder+"HO Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefHF=getHisto(folder+"HF Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefSimp=getHisto(folder+"SIPM Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefZDC=getHisto(folder+"ZDC Pedestal Reference RMS Distribution (individual cap)", process_, dbe_, debug_,cloneME_);
  PedestalsRmsRefHB->SetLineColor(kGreen);
  PedestalsRmsRefHE->SetLineColor(kGreen);
  PedestalsRmsRefHO->SetLineColor(kGreen);
  PedestalsRmsRefHF->SetLineColor(kGreen);
  PedestalsRmsRefSimp->SetLineColor(kGreen);
  PedestalsRmsRefZDC->SetLineColor(kGreen);
  PedestalsRmsRefHB->SetLineWidth(3);
  PedestalsRmsRefHE->SetLineWidth(3);
  PedestalsRmsRefHO->SetLineWidth(3);
  PedestalsRmsRefHF->SetLineWidth(3);
  PedestalsRmsRefSimp->SetLineWidth(3);
  PedestalsRmsRefZDC->SetLineWidth(3);
 
  PedestalsRmsHBref=getHisto(folder+"HB Pedestal_rms-Reference_rms Distribution", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHEref=getHisto(folder+"HE Pedestal_rms-Reference_rms Distribution", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHOref=getHisto(folder+"HO Pedestal_rms-Reference_rms Distribution", process_, dbe_, debug_,cloneME_);
  PedestalsRmsHFref=getHisto(folder+"HF Pedestal_rms-Reference_rms Distribution", process_, dbe_, debug_,cloneME_);
    
  getSJ6histos("HcalDetDiagPedestalMonitor/channel status/","Channel Status Missing Channels", ChannelStatusMissingChannels);
  getSJ6histos("HcalDetDiagPedestalMonitor/channel status/","Channel Status Unstable Channels",ChannelStatusUnstableChannels);
  getSJ6histos("HcalDetDiagPedestalMonitor/channel status/","Channel Status Pedestal Mean",    ChannelStatusBadPedestalMean);
  getSJ6histos("HcalDetDiagPedestalMonitor/channel status/","Channel Status Pedestal RMS",     ChannelStatusBadPedestalRMS);
 
  MonitorElement* me = dbe_->get("Hcal/HcalDetDiagPedestalMonitor/HcalDetDiagPedestalMonitor Event Number");
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  me = dbe_->get("Hcal/HcalDetDiagPedestalMonitor/HcalDetDiagPedestalMonitor Reference Run");
  if(me) {
    string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
  }
} 
bool HcalDetDiagPedestalClient::haveOutput(){
    getHistograms();
    if(ievt_>100) return true;
    return false; 
}
int  HcalDetDiagPedestalClient::SummaryStatus(){
    if(status==0) return 0;
    if(status==1) return 1;
    return 2;
}
double HcalDetDiagPedestalClient::get_channel_status(std::string subdet,int eta,int phi,int depth,int type){
   int ind=-1;
   if(subdet.compare("HB")==0 || subdet.compare("HF")==0) if(depth==1) ind=0; else ind=1;
   else if(subdet.compare("HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(subdet.compare("HO")==0) ind=3; 
   if(ind==-1) return -1.0;
   if(type==1) return ChannelStatusMissingChannels[ind] ->GetBinContent(eta+42,phi+1);
   if(type==2) return ChannelStatusUnstableChannels[ind]->GetBinContent(eta+42,phi+1);
   if(type==3) return ChannelStatusBadPedestalMean[ind] ->GetBinContent(eta+42,phi+1);
   if(type==4) return ChannelStatusBadPedestalRMS[ind]  ->GetBinContent(eta+42,phi+1);
   return -1.0;
}

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
void HcalDetDiagPedestalClient::htmlOutput(int runNo, string htmlDir, string htmlName){
int  MissingCnt=0;
int  UnstableCnt=0;
int  BadCnt=0; 
int  HBP[4]={0,0,0,0}; 
int  HBM[4]={0,0,0,0}; 
int  HEP[4]={0,0,0,0}; 
int  HEM[4]={0,0,0,0}; 
int  HFP[4]={0,0,0,0}; 
int  HFM[4]={0,0,0,0}; 
int  HO[4] ={0,0,0,0}; 
int  newHBP[4]={0,0,0,0}; 
int  newHBM[4]={0,0,0,0}; 
int  newHEP[4]={0,0,0,0}; 
int  newHEM[4]={0,0,0,0}; 
int  newHFP[4]={0,0,0,0}; 
int  newHFM[4]={0,0,0,0}; 
int  newHO[4] ={0,0,0,0}; 
std::string subdet[4]={"HB","HE","HO","HF"};

  HcalLogicalMapGenerator gen;
  HcalLogicalMap lmap(gen.createMap());
  HcalElectronicsMap emap=lmap.generateHcalElectronicsMap();
  
  // check how many problems we have:
  for(int sd=0;sd<4;sd++)
    {
      int feta=0,teta=0,fdepth=0,tdepth=0; 
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2; } 
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3; } 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4; } 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2; } 
      for(int phi=1;phi<=72;phi++) 
	{
	  for(int depth=fdepth;depth<=tdepth;depth++) 
	    {
	      for(int eta=feta;eta<=teta;eta++)
		{
		  if(sd==3 && eta>-29 && eta<29) continue;
		  double problem[4]={0,0,0,0};
		  problem[0] =get_channel_status(subdet[sd],eta,phi,depth,1); 
		  problem[1] =get_channel_status(subdet[sd],eta,phi,depth,2);
		  problem[2] =get_channel_status(subdet[sd],eta,phi,depth,3);
		  problem[3] =get_channel_status(subdet[sd],eta,phi,depth,4);
		  for(int i=0;i<4;i++){
		    if(problem[i]>0){
		      if(sd==0){
			if(eta>0)
			  { 
			    HBP[i]++; 
			    if(IsKnownBadChannel("HB",eta,phi,depth)!=(i+1)) newHBP[i]++;
			  }
			else
			  {
			    HBM[i]++; 
			    if(IsKnownBadChannel("HB",eta,phi,depth)!=(i+1)) newHBM[i]++;
			  } 
		      } // if (sd==0)
		      if(sd==1){
			if(eta>0)
			  { 
			    HEP[i]++; 
			    if(IsKnownBadChannel("HE",eta,phi,depth)!=(i+1)) newHEP[i]++;
			  }
			else{
			  HEM[i]++; 
			  if(IsKnownBadChannel("HE",eta,phi,depth)!=(i+1)) newHEM[i]++;}
		      } // if (sd==1)
		      if(sd==2)         { HO[i]++;  if(IsKnownBadChannel("HO",eta,phi,depth)!=(i+1)) newHO[i]++; }
		      if(sd==3){
			if(eta>0){
			  HFP[i]++; if(IsKnownBadChannel("HF",eta,phi,depth)!=(i+1)) newHFP[i]++;}
			else{
			  HFM[i]++; if(IsKnownBadChannel("HF",eta,phi,depth)!=(i+1)) newHFM[i]++;}
		      } // if (sd==3)
		    } // if (problem[i]>0)
		  } //for (int i=0;i<4;i++)
		} // for (int eta=feta;...)
	    } // for (int depth=fdepth;...)
	} // for (int phi = 1;...)
    }//for (int sd=0;...
  
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
	 std::string comm="   ";
	 if(IsKnownBadChannel(subdet[sd],eta,phi,depth)==1) comm="KNOWN PROBLEM";
         if(missing>0){
            try{
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(Missing,cnt++,*detid,lmap_entry,emap_entry,comm); MissingCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(Missing);
  Missing.close();

  // unstable channels list
  ofstream Unstable;
  Unstable.open((htmlDir + "Unstable_"+htmlName).c_str());
  printTableHeader(Unstable,"Unstable Channels list");
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
	       char comment[100]; sprintf(comment,"Missing in %.3f%% of events\n",(1.0-unstable)*100.0);
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
  
  // bad pedestals channels list
  ofstream Bad;
  Bad.open((htmlDir + "Bad_"+htmlName).c_str());
  printTableHeader(Bad,"Bad pedestal/rms Channels list");
  for(int sd=0;sd<4;sd++){
      int cnt=0;
      if(sd==0 && (HBM[2]+HBP[2])==0 && (HBM[3]+HBP[3])==0) continue;
      if(sd==1 && (HEM[2]+HEP[2])==0 && (HEM[3]+HEP[3])==0) continue;
      if(sd==2 &&  (HO[2])==0 && (HO[3])==0)                continue;
      if(sd==3 && (HFM[2]+HFP[2])==0 && (HFM[3]+HFP[3])==0) continue;
      Bad << "<tr><td align=\"center\"><h3>"<< subdet[sd] <<"</h3></td></tr>" << endl;
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4;} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2;} 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double bad1 =get_channel_status(subdet[sd],eta,phi,depth,3);
         double bad2 =get_channel_status(subdet[sd],eta,phi,depth,4);
	 std::string comm="";
	 if(IsKnownBadChannel(subdet[sd],eta,phi,depth)==2 || IsKnownBadChannel(subdet[sd],eta,phi,depth)==3) comm=", KNOWN PROBLEM";
         if(bad1>0 || bad2>0){
	   try{
	     char comment[100]; 
	       if(bad1>0) sprintf(comment,"|Ped-Ref|=%.2f%s\n",bad1,comm.c_str());
	       if(bad2>0) sprintf(comment,"|Rms-Ref|=%.2f%s\n",bad2,comm.c_str());
	       if(bad1>0 && bad2>0) sprintf(comment,"|Ped-Ref|=%.2f,|Rms-Ref|=%.2f%s\n",bad1,bad2,comm.c_str());
	       HcalDetId *detid=0;
               if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
               if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
               if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
               if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	       HcalFrontEndId    lmap_entry=lmap.getHcalFrontEndId(*detid);
	       HcalElectronicsId emap_entry=emap.lookup(*detid);
	       printTableLine(Bad,cnt++,*detid,lmap_entry,emap_entry,comment); BadCnt++;
	       delete detid;
	    }catch(...){ continue;}
         }
      }	
  }
  printTableTail(Bad);
  Bad.close();
  
  if (debug_>0) cout << "<HcalDetDiagPedestalClient::htmlOutput> Preparing  html output ..." << endl;
  if(!dbe_) return;
  string client = "HcalDetDiagPedestalClient";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
  TPaveStats *ptstats;
  TCanvas *can=new TCanvas("HcalDetDiagPedestalClient","HcalDetDiagPedestalClient",0,0,500,350);
  can->cd();
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());
  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Detector Diagnostics Pedestal Monitor</title> " << endl;
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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics Pedestal Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  /////////////////////////////////////////// 
  htmlFile << "<table width=100% border=1>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">SebDet</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Missing</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Unstable</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad |Ped-Ref|</td>" << endl;
  htmlFile << "<td class=\"s0\" width=20% align=\"center\">Bad |Rms-Ref|</td>" << endl;
  htmlFile << "</tr><tr>" << endl;
  int ind1=0,ind2=0,ind3=0,ind4=0;
  htmlFile << "<td class=\"s1\" align=\"center\">HB+</td>" << endl;
  ind1=3; if(newHBP[0]==0) ind1=2; if(newHBP[0]>0 && newHBP[0]<=12) ind1=1; if(newHBP[0]>=12 && newHBP[0]<1296) ind1=0; 
  ind2=3; if(newHBP[1]==0) ind2=2; if(newHBP[1]>0)  ind2=1; if(newHBP[1]>21)  ind2=0; 
  ind3=3; if(newHBP[2]==0) ind3=2; if(newHBP[2]>0)  ind3=1; if(newHBP[2]>21)  ind3=0;
  ind4=3; if(newHBP[3]==0) ind4=2; if(newHBP[3]>0)  ind4=1; if(newHBP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;  
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1) status|=1; 
  htmlFile << state[ind1] << HBP[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HBP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HBP[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HBP[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HB-</td>" << endl;
  ind1=3; if(newHBM[0]==0) ind1=2; if(newHBM[0]>0 && newHBM[0]<=12) ind1=1; if(newHBM[0]>=12 && newHBM[0]<1296) ind1=0; 
  ind2=3; if(newHBM[1]==0) ind2=2; if(newHBM[1]>0)  ind2=1; if(newHBM[1]>21)  ind2=0; 
  ind3=3; if(newHBM[2]==0) ind3=2; if(newHBM[2]>0)  ind3=1; if(newHBM[2]>21)  ind3=0;
  ind4=3; if(newHBM[3]==0) ind4=2; if(newHBM[3]>0)  ind4=1; if(newHBM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HBM[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HBM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HBM[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HBM[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE+</td>" << endl;
  ind1=3; if(newHEP[0]==0) ind1=2; if(newHEP[0]>0 && newHEP[0]<=12) ind1=1; if(newHEP[0]>=12 && newHEP[0]<1296) ind1=0; 
  ind2=3; if(newHEP[1]==0) ind2=2; if(newHEP[1]>0)  ind2=1; if(newHEP[1]>21)  ind2=0; 
  ind3=3; if(newHEP[2]==0) ind3=2; if(newHEP[2]>0)  ind3=1; if(newHEP[2]>21)  ind3=0;
  ind4=3; if(newHEP[3]==0) ind4=2; if(newHEP[3]>0)  ind4=1; if(newHEP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HEP[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HEP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HEP[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HEP[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HE-</td>" << endl;
  ind1=3; if(newHEM[0]==0) ind1=2; if(newHEM[0]>0 && newHEM[0]<=12) ind1=1; if(newHEM[0]>=12 && newHEM[0]<1296) ind1=0; 
  ind2=3; if(newHEM[1]==0) ind2=2; if(newHEM[1]>0)  ind2=1; if(newHEM[1]>21)  ind2=0; 
  ind3=3; if(newHEM[2]==0) ind3=2; if(newHEM[2]>0)  ind3=1; if(newHEM[2]>21)  ind3=0;
  ind4=3; if(newHEM[3]==0) ind4=2; if(newHEM[3]>0)  ind4=1; if(newHEM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HEM[0] <<" (1296)</td>" << endl;
  htmlFile << state[ind2] << HEM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HEM[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HEM[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF+</td>" << endl;
  ind1=3; if(newHFP[0]==0) ind1=2; if(newHFP[0]>0 && newHFP[0]<=12) ind1=1; if(newHFP[0]>=12 && newHFP[0]<864) ind1=0; 
  ind2=3; if(newHFP[1]==0) ind2=2; if(newHFP[1]>0)  ind2=1; if(newHFP[1]>21)  ind2=0; 
  ind3=3; if(newHFP[2]==0) ind3=2; if(newHFP[2]>0)  ind3=1; if(newHFP[2]>21)  ind3=0;
  ind4=3; if(newHFP[3]==0) ind4=2; if(newHFP[3]>0)  ind4=1; if(newHFP[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HFP[0] <<" (864)</td>" << endl;
  htmlFile << state[ind2] << HFP[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HFP[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HFP[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HF-</td>" << endl;
  ind1=3; if(newHFM[0]==0) ind1=2; if(newHFM[0]>0 && newHFM[0]<=12) ind1=1; if(newHFM[0]>=12 && newHFM[0]<864) ind1=0; 
  ind2=3; if(newHFM[1]==0) ind2=2; if(newHFM[1]>0)  ind2=1; if(newHFM[1]>21)  ind2=0; 
  ind3=3; if(newHFM[2]==0) ind3=2; if(newHFM[2]>0)  ind3=1; if(newHFM[2]>21)  ind3=0;
  ind4=3; if(newHFM[3]==0) ind4=2; if(newHFM[3]>0)  ind4=1; if(newHFM[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HFM[0] <<" (864)</td>" << endl;
  htmlFile << state[ind2] << HFM[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HFM[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HFM[3] <<"</td>" << endl;
  
  htmlFile << "</tr><tr>" << endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HO</td>" << endl;
  ind1=3; if(newHO[0]==0) ind1=2; if(newHO[0]>0 && newHO[0]<=12) ind1=1; if(newHO[0]>=12 && newHO[0]<2160) ind1=0; 
  ind2=3; if(newHO[1]==0) ind2=2; if(newHO[1]>0)  ind2=1; if(newHO[1]>21)  ind2=0; 
  ind3=3; if(newHO[2]==0) ind3=2; if(newHO[2]>0)  ind3=1; if(newHO[2]>21)  ind3=0;
  ind4=3; if(newHO[3]==0) ind4=2; if(newHO[3]>0)  ind4=1; if(newHO[3]>21)  ind4=0;
  if(ind1==3) ind2=ind3=ind4=3;
  if(ind1==0 || ind2==0 || ind3==0 || ind4==0) status|=2; else if(ind1==1 || ind2==1 || ind3==1 || ind4==1)status|=1; 
  htmlFile << state[ind1] << HO[0] <<" (2160)</td>" << endl;
  htmlFile << state[ind2] << HO[1] <<"</td>" << endl;
  htmlFile << state[ind3] << HO[2] <<"</td>" << endl;
  htmlFile << state[ind4] << HO[3] <<"</td>" << endl;
  
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  /////////////////////////////////////////// 
  if((MissingCnt+UnstableCnt+BadCnt)>0){
      htmlFile << "<table width=100% border=1><tr>" << endl;
      if(MissingCnt>0)  htmlFile << "<td><a href=\"" << ("Missing_"+htmlName).c_str() <<"\">list of missing channels</a></td>";
      if(UnstableCnt>0) htmlFile << "<td><a href=\"" << ("Unstable_"+htmlName).c_str() <<"\">list of unstable channels</a></td>";
      if(BadCnt>0)      htmlFile << "<td><a href=\"" << ("Bad_"+htmlName).c_str() <<"\">list of bad pedestal/rms channels</a></td>";
      htmlFile << "</tr></table>" << endl;
  }
  can->SetGridy();
  can->SetGridx();
  can->SetLogy(0); 
  
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">Summary plots</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Pedestals2DHBHEHF->SetStats(0);
  Pedestals2DHBHEHF->SetMaximum(5);
  Pedestals2DHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_pedestal_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_pedestal_map.gif\" alt=\"hbhehf pedestal mean map\">   </td>" << endl;
  Pedestals2DHO->SetStats(0);
  Pedestals2DHO->SetMaximum(5);
  Pedestals2DHO->SetNdivisions(36,"Y");
  Pedestals2DHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_pedestal_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_map.gif\" alt=\"ho pedestal mean map\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  Pedestals2DRmsHBHEHF->SetStats(0);
  Pedestals2DRmsHBHEHF->SetMaximum(2);
  Pedestals2DRmsHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DRmsHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_rms_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_rms_map.gif\" alt=\"hbhehf pedestal rms map\">   </td>" << endl;
  Pedestals2DRmsHO->SetStats(0);
  Pedestals2DRmsHO->SetMaximum(2);
  Pedestals2DRmsHO->SetNdivisions(36,"Y");
  Pedestals2DRmsHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_rms_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_rms_map.gif\" alt=\"ho pedestal rms map\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  Pedestals2DErrorHBHEHF->SetStats(0);
  Pedestals2DErrorHBHEHF->SetNdivisions(36,"Y");
  Pedestals2DErrorHBHEHF->Draw("COLZ");
  can->SaveAs((htmlDir + "hbhehf_error_map.gif").c_str());
  htmlFile << "<td><img src=\"hbhehf_error_map.gif\" alt=\"hbhehf pedestal error map\">   </td>" << endl;
  Pedestals2DErrorHO->SetStats(0);
  Pedestals2DErrorHO->SetNdivisions(36,"Y");
  Pedestals2DErrorHO->Draw("COLZ");
  can->SaveAs((htmlDir + "ho_error_map.gif").c_str());
  htmlFile << "<td><img src=\"ho_error_map.gif\" alt=\"ho pedestal error map\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<hr>" << endl;
  
  ///////////////////////////////////////////   
  htmlFile << "<h2 align=\"center\">HB Pedestal plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsRefAve4HB->GetMaximum()>0 && PedestalsAve4HB->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsAve4HB->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRefAve4HB->GetListOfFunctions()->Add(ptstats);
  PedestalsRefAve4HB->Draw();
  PedestalsAve4HB->Draw("Sames");
  can->SaveAs((htmlDir + "hb_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_distribution.gif\" alt=\"hb pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefHB->GetMaximum()>0 && PedestalsRmsHB->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsRmsHB->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRmsRefHB->GetListOfFunctions()->Add(ptstats);
  PedestalsRmsRefHB->Draw();
  PedestalsRmsHB->Draw("Sames");
  can->SaveAs((htmlDir + "hb_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_rms_distribution.gif\" alt=\"hb pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsAve4HBref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HBref->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_ref_distribution.gif\" alt=\"hb pedestal-reference mean\">   </td>" << endl;
  if(PedestalsRmsHBref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHBref->Draw();
  can->SaveAs((htmlDir + "hb_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hb_pedestal_rms_ref_distribution.gif\" alt=\"hb pedestal rms-reference mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">HE Pedestal plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsRefAve4HE->GetMaximum()>0 && PedestalsAve4HE->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0);  
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsAve4HE->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRefAve4HE->GetListOfFunctions()->Add(ptstats);
  PedestalsRefAve4HE->Draw();
  PedestalsAve4HE->Draw("Sames");
  can->SaveAs((htmlDir + "he_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_distribution.gif\" alt=\"he pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefHE->GetMaximum()>0 && PedestalsRmsHE->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsRmsHE->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRmsRefHE->GetListOfFunctions()->Add(ptstats);
  PedestalsRmsRefHE->Draw();
  PedestalsRmsHE->Draw("Sames");
  can->SaveAs((htmlDir + "he_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_rms_distribution.gif\" alt=\"he pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsAve4HEref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HEref->Draw();
  can->SaveAs((htmlDir + "he_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_ref_distribution.gif\" alt=\"he pedestal-reference mean\">   </td>" << endl;
  if(PedestalsRmsHEref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHEref->Draw();
  can->SaveAs((htmlDir + "he_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"he_pedestal_rms_ref_distribution.gif\" alt=\"he pedestal rms-reference mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">HO Pedestal plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsRefAve4HO->GetMaximum()>0 && PedestalsAve4HO->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsAve4HO->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRefAve4HO->GetListOfFunctions()->Add(ptstats);
  PedestalsRefAve4HO->Draw();
  PedestalsAve4HO->Draw("Sames");
  can->SaveAs((htmlDir + "ho_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_distribution.gif\" alt=\"ho pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefHO->GetMaximum()>0 && PedestalsRmsHO->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsRmsHO->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRmsRefHO->GetListOfFunctions()->Add(ptstats);  PedestalsRmsRefHO->Draw();
  PedestalsRmsHO->Draw("Sames");
  can->SaveAs((htmlDir + "ho_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_rms_distribution.gif\" alt=\"ho pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  // SIMP
   htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsRefAve4Simp->GetMaximum()>0 && PedestalsAve4Simp->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsAve4Simp->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRefAve4Simp->GetListOfFunctions()->Add(ptstats);
  PedestalsRefAve4Simp->Draw();
  PedestalsAve4Simp->Draw("Sames");
  can->SaveAs((htmlDir + "sipm_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"sipm_pedestal_distribution.gif\" alt=\"sipm pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefSimp->GetMaximum()>0 && PedestalsRmsSimp->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsRmsSimp->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRmsRefSimp->GetListOfFunctions()->Add(ptstats); 
  PedestalsRmsRefSimp->Draw();
  PedestalsRmsSimp->Draw("Sames");
  can->SaveAs((htmlDir + "simp_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"simp_pedestal_rms_distribution.gif\" alt=\"sipm pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  // SIMP
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsAve4HOref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HOref->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_ref_distribution.gif\" alt=\"ho pedestal-reference mean\">   </td>" << endl;
  if(PedestalsRmsHOref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHOref->Draw();
  can->SaveAs((htmlDir + "ho_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"ho_pedestal_rms_ref_distribution.gif\" alt=\"ho pedestal rms-reference mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">HF Pedestal plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsRefAve4HF->GetMaximum()>0 && PedestalsAve4HF->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsAve4HF->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRefAve4HF->GetListOfFunctions()->Add(ptstats);
  PedestalsRefAve4HF->Draw();
  PedestalsAve4HF->Draw("Sames");
  can->SaveAs((htmlDir + "hf_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_distribution.gif\" alt=\"hf pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefHF->GetMaximum()>0 && PedestalsRmsHF->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
  ptstats->SetTextColor(kBlack);
  PedestalsRmsHF->GetListOfFunctions()->Add(ptstats);
  ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
  ptstats->SetTextColor(kGreen);
  PedestalsRmsRefHF->GetListOfFunctions()->Add(ptstats);  
  PedestalsRmsRefHF->Draw();
  PedestalsRmsHF->Draw("Sames");
  can->SaveAs((htmlDir + "hf_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_rms_distribution.gif\" alt=\"hf pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  if(PedestalsAve4HFref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsAve4HFref->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_ref_distribution.gif\" alt=\"hf pedestal-reference mean\">   </td>" << endl;
  if(PedestalsRmsHFref->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
  PedestalsRmsHFref->Draw();
  can->SaveAs((htmlDir + "hf_pedestal_rms_ref_distribution.gif").c_str());
  htmlFile << "<td><img src=\"hf_pedestal_rms_ref_distribution.gif\" alt=\"hf pedestal rms-reference mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  
  /////////////////////////////////////////// 
  htmlFile << "<h2 align=\"center\">ZDC Pedestal plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  
  if(PedestalsRefAve4ZDC->GetMaximum()>0 && PedestalsAve4ZDC->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
//   ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
//   ptstats->SetTextColor(kBlack);
//   PedestalsAve4ZDC->GetListOfFunctions()->Add(ptstats);
//   ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
//   ptstats->SetTextColor(kGreen);
//   PedestalsRefAve4ZDC->GetListOfFunctions()->Add(ptstats);
//   PedestalsRefAve4ZDC->Draw();
//   PedestalsAve4ZDC->Draw("Sames");
  PedestalsAve4ZDC->Draw();
  can->SaveAs((htmlDir + "zdc_pedestal_distribution.gif").c_str());
  htmlFile << "<td><img src=\"zdc_pedestal_distribution.gif\" alt=\"zdc pedestal mean\">   </td>" << endl;
  if(PedestalsRmsRefZDC->GetMaximum()>0 && PedestalsRmsZDC->GetMaximum()>0) can->SetLogy(1); else can->SetLogy(0); 
//   ptstats = new TPaveStats(0.75,0.8,0.99,1.0,"brNDC");
//   ptstats->SetTextColor(kBlack);
//   PedestalsRmsZDC->GetListOfFunctions()->Add(ptstats);
//   ptstats = new TPaveStats(0.75,0.58,0.99,0.78,"brNDC");
//   ptstats->SetTextColor(kGreen);
//   PedestalsRmsRefZDC->GetListOfFunctions()->Add(ptstats); 
//   PedestalsRmsRefZDC->Draw();
//   PedestalsRmsZDC->Draw("Sames");
  PedestalsRmsZDC->Draw();
  can->SaveAs((htmlDir + "zdc_pedestal_rms_distribution.gif").c_str());
  htmlFile << "<td><img src=\"zdc_pedestal_rms_distribution.gif\" alt=\"zdc pedestal rms mean\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();
} 


