#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorLEDMonitor ***********************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 20.11.2008 (first version) ******// 
//---------- last revision: 31.05.2011 (Panos Katsas) 
//***************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//
//======================= Constructor ==============================//
CastorLEDMonitor::CastorLEDMonitor(const edm::ParameterSet& ps)
{
 subsystemname =
        ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 ievt_=0;
}

//======================= Destructor ==============================//
CastorLEDMonitor::~CastorLEDMonitor() { }
  
//========================= setup ==========================//
void CastorLEDMonitor::setup(const edm::ParameterSet& ps)
{
  CastorBaseMonitor::setup(ps);
  ievt_=0;
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::setup (end)"<<std::endl;  
  return;
}

//============= bookHistograms =================//
void CastorLEDMonitor::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  char s[60];
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::bookHistograms"<<std::endl;  

  ibooker.setCurrentFolder(subsystemname + "/CastorLEDMonitor");
  sprintf(s,"CastorLED_qVsTS(allPMT)");
   //h2qts = ibooker.book2D(s,s, 10,0,10., 5000,0.,10000.);
   //h2qts->getTH2F()->GetXaxis()->SetTitle("TS");
   //h2qts->getTH2F()->GetYaxis()->SetTitle("Qcastor(fC)");
   //h2qts->getTH2F()->SetOption("colz");

  sprintf(s,"CastorLED_qVsPMT");
   //h2QvsPMT = ibooker.book2D(s,s, 224,0,224, 5000,0.,50000.);    
   //h2QvsPMT->getTH2F()->GetXaxis()->SetTitle("sector*14+module");
   //h2QvsPMT->getTH2F()->GetYaxis()->SetTitle("RecHit");
   //h2QvsPMT->getTH2F()->SetOption("colz");

  sprintf(s,"CastorLEDqMap(cumulative)");
    h2qMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2qMap->getTH2F()->SetOption("colz");
  sprintf(s,"CastorLED_QmeanMap");
    h2meanMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2meanMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
    h2meanMap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2meanMap->getTH2F()->SetOption("colz");

 ievt_=0;
 if(fVerbosity>0) std::cout<<"CastorLEDMonitor::beginRun(end)"<<std::endl; 
 return;
}

//=================== processEvent  ========================//
void CastorLEDMonitor::processEvent( const CastorDigiCollection& castorDigis, const CastorDbService& cond)
  {
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::processEvent (start)"<<std::endl;  

/* be implemented
 edm::Handle<HcalTBTriggerData> trigger_data;
 iEvent.getByToken(tok_tb_, trigger_data);
 if(trigger_data.isValid()) 
  if(trigger_data->triggerWord()==6) LEDevent=true; 
*/
  
 if(castorDigis.size() <= 0) {
  if(fVerbosity > 0) 
 std::cout<<"CastorLEDMonitor::processEvent NO Castor Digis"<<std::endl;
  return;
 }

 for(CastorDigiCollection::const_iterator j=castorDigis.begin(); j!=castorDigis.end(); j++)
 {
   const CastorDataFrame digi = (const CastorDataFrame)(*j);
   int module = digi.id().module()-1;
   int sector = digi.id().sector()-1;
   double qsum=0.;
   for(int i=0; i<digi.size(); i++) {
     int dig=digi.sample(i).adc() & 0x7f;
     float ets = LedMonAdc2fc[dig] + 0.5;
     //h2qts->Fill(i,ets);
     qsum += ets;
   }
   //int ind = sector*14 + module;
   //h2QvsPMT->Fill(ind,qsum);
   h2qMap->Fill(module,sector,qsum);
 } // end for(CastorDigiCollection::const_iterator j=castorDigis...

  ievt_++; 
  if(ievt_%100 == 0) {
   for(int mod=1; mod<=14; mod++) for(int sec=1; sec<=16;sec++) {
    double a= h2qMap->getTH2F()->GetBinContent(mod,sec);
    h2meanMap->getTH2F()->SetBinContent(mod,sec,a/double(ievt_));
   }
  }

//if(fVerbosity>0) std::cout<<"CastorLEDMonitor::processEvent(end)"<<std::endl;
  return;
}
