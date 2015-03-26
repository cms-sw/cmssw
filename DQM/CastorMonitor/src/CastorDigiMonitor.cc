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
//==================================================================//
//======================= Constructor ==============================//
CastorDigiMonitor::CastorDigiMonitor(const edm::ParameterSet& ps)
{
subsystemname_=ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 fVerbosity = ps.getUntrackedParameter<int>("debug",0);
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
  ievt_=0;
//   pset_.getUntrackedParameter<std::string>("subSystemFolder","Castor");
//  baseFolder_ = rootFolder_+"CastorDigiMonitor";

  ibooker.setCurrentFolder(subsystemname_ + "/CastorDigiMonitor");
	std::string s2 = "QIE_capID+er+dv";
    	h2digierr = ibooker.book2D(s2,s2,14,0.,14., 16,0.,16.);
	h2digierr->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
	h2digierr->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
	h2digierr->getTH2F()->SetOption("colz");

  sprintf(s,"BunchOccupancy(fC)_all_TS");
    hBunchOcc = ibooker.bookProfile(s,s,4000,0,4000, 100,0,1.e10,"");
    hBunchOcc->getTProfile()->GetXaxis()->SetTitle("BX");
    hBunchOcc->getTProfile()->GetYaxis()->SetTitle("QIE(fC)");

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
    sprintf(s,"QfC=f(Tile TS) (cumulative)");
       h2QtsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
  h2QtsvsCh->getTH2F()->GetXaxis()->SetTitle("Tile(=sector*14+module)");
       h2QtsvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
       h2QtsvsCh->getTH2F()->SetOption("colz");
    sprintf(s,"QmeanfC=f(Tile TS)");
      h2QmeantsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
  h2QmeantsvsCh->getTH2F()->GetXaxis()->SetTitle("Tile(=sector*14+module)");
      h2QmeantsvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
      h2QmeantsvsCh->getTH2F()->SetOption("colz");
    sprintf(s,"QmeanfC_map(allTS)");
      h2QmeanMap = ibooker.book2D(s,s,14,0.,14., 16,0.,16.);
      h2QmeanMap->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
      h2QmeanMap->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
      h2QmeanMap->getTH2F()->SetOption("textcolz");

 if(fVerbosity>0) std::cout<<"CastorDigiMonitor::beginRun(end)"<<std::endl;
 return;
}


//=============== processEvent  =========
void CastorDigiMonitor::processEvent(const CastorDigiCollection& castorDigis,
	const CastorDbService& cond, int iBunch) {
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (begin)"<< std::endl;

//  CaloSamples tool;  
 
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
//   for (int i=1; i<digi.size(); i++) {
   for (int i=0; i<digi.size(); i++) {
     int module = digi.id().module()-1;
     int sector = digi.id().sector()-1;
     if(capid1 < 3) capid1++;
     else capid1 = 0;
     int capid = digi.sample(i).capid();
     int dv = digi.sample(i).dv();
     int er = digi.sample(i).er();
     int rawd = digi.sample(i).adc();
     rawd = rawd&0x7F;
     int err = (capid != capid1) | er<<1 | (!dv)<<2; // =0
     if(err !=0) h2digierr->Fill(module,sector);
     int ind = sector*14 + module;
     h2QtsvsCh->Fill(ind,i,LedMonAdc2fc[rawd]);
     sum += LedMonAdc2fc[rawd];
//     if(err != 0 && fVerbosity>0)
//     std::cout<<"event/idigi=" <<ievt_<<"/" <<i<< " cap_cap1_dv_er: " <<
//	capid <<"="<< capid1 <<" "<< dv <<" "<< er<<" "<< err << std::endl;
     capid1 = capid;
   }
   hBunchOcc->Fill(iBunch,sum);
 } //end for(CastorDigiCollection::const_iterator ...

 ievt_++;
 if(ievt_ %100 == 0) {
   float ModuleSum[14], SectorSum[16];
   for(int m=0; m<14; m++) ModuleSum[m]=0.;
   for(int s=0; s<16; s++) SectorSum[s]=0.;
   for(int mod=0; mod<14; mod++) for(int sec=0; sec<16; sec++) {
     double sum=0.;
     for(int ts=1; ts<=10; ts++) {
       int ind = sec*14 + mod +1;
       double a=h2QtsvsCh->getTH2F()->GetBinContent(ind,ts);
  h2QmeantsvsCh->getTH2F()->SetBinContent(ind,ts,a/double(ievt_));
       sum += a;
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
 } //end if(ievt_ %100 == 0) {

  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (end)"<< std::endl;
  return;
 }
