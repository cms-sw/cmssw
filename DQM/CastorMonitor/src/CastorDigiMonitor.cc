#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//****************************************************//
//********** CastorDigiMonitor: ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 29.08.2008 (first version) ******// 
////---- digi values in Castor r/o channels 
//// last revision: 31.05.2011 (Panos Katsas) to remove selecting N events for filling the histograms
//****************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//     add rms check     15.04.2015 (Vladimir Popov)
//==================================================================//

 static int TS_MAX = 10;
 static float RatioThresh1 = 0.;
 static float QIEerrThreshold = 0.0001;
 static double QrmsTS[224][10], QmeanTS[224][10];
 const int TSped = 0;

//======================= Constructor ==============================//
CastorDigiMonitor::CastorDigiMonitor(const edm::ParameterSet& ps)
{
 fVerbosity = ps.getUntrackedParameter<int>("debug",0);
subsystemname_=ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 RatioThresh1 = ps.getUntrackedParameter<double>("ratioThreshold",0.9);
 Qrms_DEAD = ps.getUntrackedParameter<double>("QrmsDead",0.01); //fC
 Qrms_DEAD = Qrms_DEAD*Qrms_DEAD;
 TS_MAX = ps.getUntrackedParameter<double>("qieTSmax",6);
}

//======================= Destructor ===============================//
CastorDigiMonitor::~CastorDigiMonitor() { }

//=========================== setup  ===============//
void CastorDigiMonitor::setup(const edm::ParameterSet& ps)
{
 subsystemname_=
   ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
  CastorBaseMonitor::setup(ps);
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::setup (end)"<<std::endl;
  return;
}

//================= bookHistograms ===================//
void CastorDigiMonitor::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  char s[60];
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::beginRun (start)" << std::endl;
  char sTileIndex[50];
  sprintf(sTileIndex,"Tile(=module*16+sector)");

  ievt_=0;

  ibooker.setCurrentFolder(subsystemname_ + "/CastorDigiMonitor");

  std::string s2 = "CASTOR QIE_capID+er+dv";
  h2digierr=ibooker.bookProfile2D(s2,s2,14,0.,14., 16,0.,16.,100,0,1.e10,"");
  h2digierr->getTProfile2D()->GetXaxis()->SetTitle("ModuleZ");
  h2digierr->getTProfile2D()->GetYaxis()->SetTitle("SectorPhi");
  h2digierr->getTProfile2D()->SetMaximum(1.);
  h2digierr->getTProfile2D()->SetMinimum(QIEerrThreshold);
  h2digierr->getTProfile2D()->SetOption("colz");

  sprintf(s,"CASTOR DeadChannelsMap");
    h2status = ibooker.book2D(s,s,14,0.,14., 16,0.,16.);   
    h2status->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
    h2status->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
    h2status->getTH2F()->SetOption("colz");

  sprintf(s,"CASTOR AverageToMaxRatioMap");
    h2TSratio = ibooker.book2D(s,s,14,0.,14., 16,0.,16.);   
    h2TSratio->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
    h2TSratio->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
    h2TSratio->getTH2F()->SetOption("colz");

  sprintf(s,"CASTOR AverageToMaxRatio");
    hTSratio = ibooker.book1D(s,s,100,0.,1.);   

    sprintf(s,"DigiSize");
        hdigisize = ibooker.book1D(s,s,20,0.,20.);
    sprintf(s,"Module(fC)_allTS");
        hModule = ibooker.book1D(s,s,14,0.,14.);
	hModule->getTH1F()->GetXaxis()->SetTitle("ModuleZ");
	hModule->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");
    sprintf(s,"Sector(fC)_allTS");
        hSector = ibooker.book1D(s,s,16,0.,16.);
	hSector->getTH1F()->GetXaxis()->SetTitle("SectorPhi");
	hSector->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");

    sprintf(s,"QfC=f(x=Tile y=TS) (cumulative)");
      h2QtsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
      h2QtsvsCh->getTH2F()->GetXaxis()->SetTitle(sTileIndex);
      h2QtsvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
      h2QtsvsCh->getTH2F()->SetOption("colz");

    sprintf(s,"QmeanfC=f(Tile TS)");
      h2QmeantsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
      h2QmeantsvsCh->getTH2F()->GetXaxis()->SetTitle(sTileIndex);
      h2QmeantsvsCh->getTH2F()->GetYaxis()->SetTitle("Time Slice");
      h2QmeantsvsCh->getTH2F()->SetOption("colz");

    sprintf(s,"QrmsfC=f(Tile TS)");
      h2QrmsTSvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
      h2QrmsTSvsCh->getTH2F()->GetXaxis()->SetTitle(sTileIndex);
      h2QrmsTSvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
      h2QrmsTSvsCh->getTH2F()->SetOption("colz");

 sprintf(s,"CASTORreportSummaryMap");
    h2reportMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2reportMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
    h2reportMap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2reportMap->getTH2F()->SetOption("colz");
    
   hReport = ibooker.bookFloat("CASTOR reportSummary");

 sprintf(s,"QmeanfC_map(allTS)");
      h2QmeanMap = ibooker.book2D(s,s,14,0.,14., 16,0.,16.);
      h2QmeanMap->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
      h2QmeanMap->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
      h2QmeanMap->getTH2F()->SetOption("textcolz");

 for(int ts=0; ts<=1; ts++) {
   sprintf(s,"QIErms_TS=%d",ts);
   hQIErms[ts] = ibooker.book1D(s,s,1000,0.,100.);  
   hQIErms[ts]->getTH1F()->GetXaxis()->SetTitle("QIErms(fC)");
 }

 for(int ind=0; ind<224; ind++) for(int ts=0; ts<10; ts++) 
   QrmsTS[ind][ts] = QmeanTS[ind][ts]= 0.;

 if(fVerbosity>0) std::cout<<"CastorDigiMonitor::beginRun(end)"<<std::endl;
 return;
}


//=============== processEvent  =========
void CastorDigiMonitor::processEvent(const CastorDigiCollection& castorDigis,
	const CastorDbService& cond) {
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (begin)"<< std::endl;

 if(castorDigis.size() <= 0) {
  if(fVerbosity>0) std::cout<<"CastorPSMonitor::processEvent NO Castor Digis"<<std::endl;
  return;
 }

 for(CastorDigiCollection::const_iterator j=castorDigis.begin();
	 j!=castorDigis.end(); j++)
 {
   const CastorDataFrame digi = (const CastorDataFrame)(*j);	
 
   int capid1 = digi.sample(0).capid();
   hdigisize->Fill(digi.size());
   double sum = 0.;
   for (int i=0; i<digi.size(); i++) {
     int module = digi.id().module()-1;
     int sector = digi.id().sector()-1;
     int capid = digi.sample(i).capid();
     int dv = digi.sample(i).dv();
     int er = digi.sample(i).er();
     int rawd = digi.sample(i).adc();
     rawd = rawd&0x7F;
     int err = (capid != capid1) | er<<1 | (!dv)<<2; // =0
     h2digierr->Fill(module,sector,err);
//     if(err !=0) continue;
     int ind = ModSecToIndex(module,sector);
     h2QtsvsCh->Fill(ind,i,LedMonAdc2fc[rawd]);
     float q = LedMonAdc2fc[rawd];
     sum += q;  //     sum += LedMonAdc2fc[rawd];
        QrmsTS[ind][i] += (q*q);
        QmeanTS[ind][i] += q;
     if(err != 0 && fVerbosity>0)
   std::cout<<"event/idigi=" <<ievt_<<"/"<<i<<" cap=cap1_dv_er_err: "<<
	capid <<"="<< capid1 <<" "<< dv <<" "<< er<<" "<< err << std::endl;
     if(capid1 < 3) capid1 = capid+1;
     else capid1 = 0;
   }
//   hBunchOcc->Fill(iBunch,sum);
 } //end for(CastorDigiCollection::const_iterator ...

 ievt_++;

 const float repChanBAD = 0.9;
 const float repChanWarning = 0.95;
 if(ievt_ %100 != 0) return;
   float ModuleSum[14], SectorSum[16];
   for(int m=0; m<14; m++) ModuleSum[m]=0.;
   for(int s=0; s<16; s++) SectorSum[s]=0.;
   for(int mod=0; mod<14; mod++) for(int sec=0; sec<16; sec++) {
     for(int ts=0; ts<=1; ts++) {
     int ind = ModSecToIndex(mod,sec);
       double Qmean = QmeanTS[ind][ts]/ievt_;
       double Qrms = sqrt(QrmsTS[ind][ts]/ievt_ - Qmean*Qmean);
       hQIErms[ts]->Fill(Qrms);
     }

     double sum=0.;
     for(int ts=1; ts<=TS_MAX; ts++) {
     int ind = ModSecToIndex(mod,sec) + 1;
       double a=h2QtsvsCh->getTH2F()->GetBinContent(ind,ts);
  h2QmeantsvsCh->getTH2F()->SetBinContent(ind,ts,a/double(ievt_));
       sum += a;
       double Qmean = QmeanTS[ind-1][ts-1]/ievt_;
       double Qrms = QrmsTS[ind-1][ts-1]/ievt_ - Qmean*Qmean;
  h2QrmsTSvsCh->getTH2F()->SetBinContent(ind,ts,sqrt(Qrms));
     }
     sum /= double(ievt_);
     ModuleSum[mod] += sum;
     SectorSum[sec] += sum;
     float isum = float(int(sum*10.+0.5))/10.;
     h2QmeanMap->getTH2F()->SetBinContent(mod+1,sec+1,isum);
   } // end for(int mod=0; mod<14; mod++) for(int sec=0; sec<16; sec++) 

   for(int mod=0; mod<14; mod++) 
	hModule->getTH1F()->SetBinContent(mod+1,ModuleSum[mod]);
   for(int sec=0; sec<16; sec++) 
	hSector->getTH1F()->SetBinContent(sec+1,SectorSum[sec]);

  int nGoodCh = 0;
  for(int mod=0; mod<14; mod++) for(int sec=0; sec<16;sec++) {
    int ind = ModSecToIndex(mod,sec);
    double Qmean = QmeanTS[ind][TSped]/ievt_;
    double Qrms = QrmsTS[ind][TSped]/ievt_ - Qmean*Qmean;
    float ChanStatus = 0.;
    if(Qrms < Qrms_DEAD) ChanStatus = 1.;
    h2status->getTH2F()->SetBinContent(mod+1,sec+1,ChanStatus);

    int tsm = 0;
    float am=0.;
    for(int ts=0; ts<TS_MAX; ts++) {
      float a = h2QmeantsvsCh->getTH2F()->GetBinContent(ind+1,ts+1);
      if(am < a) {am = a; tsm = ts;}
    }

    double sum = 0.;
    for(int ts=0; ts<TS_MAX; ts++) if(ts != tsm)
      sum += h2QmeantsvsCh->getTH2F()->GetBinContent(ind+1,ts+1);
    float r = 0.;
    if(am > 0.) r = sum/(TS_MAX-1)/am;
    h2TSratio->getTH2F()->SetBinContent(mod+1,sec+1,r);
    hTSratio->Fill(r);
    float statusTS = 1.0;
    if(r > RatioThresh1) statusTS = repChanWarning;
    else if(r > 0.99) statusTS = repChanBAD;
    float gChanStatus = statusTS;
    if(ChanStatus > 0.) gChanStatus = repChanBAD; // RMS
 if(h2digierr->getTProfile2D()->GetBinContent(mod+1,sec+1)>QIEerrThreshold)
	gChanStatus = repChanBAD;
    h2reportMap->getTH2F()->SetBinContent(mod+1,sec+1,gChanStatus);
    if(gChanStatus > repChanBAD) ++nGoodCh;
  }
  hReport->Fill(float(nGoodCh)/224.);
  return;
 }

int CastorDigiMonitor::ModSecToIndex(int module, int sector) {
  int ind = sector + module*16;
  if(ind>223) ind=223;
  return(ind);
}
