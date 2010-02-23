#include "DQM/HcalMonitorTasks/interface/HcalNZSMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Common/interface/TriggerNames.h" 

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
//#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
//#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

HcalNZSMonitor::HcalNZSMonitor() {

} 

HcalNZSMonitor::~HcalNZSMonitor() {}

void HcalNZSMonitor::reset(){}

void HcalNZSMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
  }

}

void HcalNZSMonitor::setup(const edm::ParameterSet& ps,
				  DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  vector<string> names;
  names.push_back(string("HLT_HcalPhiSym"));
  names.push_back(string("HLT_HcalNZS_8E29"));

  triggers_=ps.getUntrackedParameter<vector<string> >("NZSMonitor_nzsHLTnames",names);
  period_=ps.getUntrackedParameter<int>("NZSMonitor_NZSeventPeriod",4096);
  
  baseFolder_ = rootFolder_+"NZSMonitor";

  if(fVerbosity) cout << "About to pushback fedUnpackList_" << endl;

  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
    {
      selFEDs_.push_back(i);
    }

  for (unsigned int i=0; i<triggers_.size(); i++) nAcc.push_back(0);

  nAndAcc=0;
  nAcc_Total=0;
  
  if (m_dbe)
    {
      string type;
      m_dbe->setCurrentFolder(baseFolder_);
      
      meFEDsizesNZS_=m_dbe->bookProfile("FED sizes","FED sizes",32,699.5,731.5,100,-1000.0,12000.0,"");
      meFEDsizesNZS_->setAxisTitle("FED number",1);
      meFEDsizesNZS_->setAxisTitle("average size (KB)",2);
      meFEDsizesNZS_->getTProfile()->SetMarkerStyle(22);
      
      meFEDsizeVsLumi_=m_dbe->bookProfile("FED size Vs lumi block number","FED size Vs lumi block number",Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000,"");
      meFEDsizeVsLumi_->setAxisTitle("lumiblock number",1);
      meFEDsizeVsLumi_->setAxisTitle("average HCAL FED size (KB)",2);
      meFEDsizeVsLumi_->getTProfile()->SetMarkerStyle(22);

      meL1evtNumber_=m_dbe->book1D("Is L1 event number multiple of NZS period","Is L1 event number multiple of NZS period",2,0,2);
      meL1evtNumber_->setBinLabel(1, "NO", 1);
      meL1evtNumber_->setBinLabel(2, "YES", 1);

      meIsUS_=m_dbe->book1D("IsUnsupressed bit","IsUnsuppressed bit",2,0,2);
      meIsUS_->setBinLabel(1,"NO",1);
      meIsUS_->setBinLabel(2,"YES",1);

      meBXtriggered_=m_dbe->book1D("Triggered BX number","Triggered BX number",3850,0,3850);
      meBXtriggered_->setAxisTitle("BX number",1);

      meTrigFrac_=m_dbe->book1D("HLT accept fractions","HLT accept fractions",triggers_.size()+1,0,triggers_.size()+1);
      for (unsigned int k=0; k<triggers_.size(); k++) meTrigFrac_->setBinLabel(k+1,triggers_[k].c_str(),1);
      meTrigFrac_->setBinLabel(triggers_.size()+1,"AND",1);

      meFullCMSdataSize_=m_dbe->bookProfile("full CMS data size","full CMS data size",Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000,"");
      meFullCMSdataSize_->setAxisTitle("lumiblock number",1);
      meFullCMSdataSize_->setAxisTitle("average FEDRawDataCollection size (KB)",2);
      meFullCMSdataSize_->getTProfile()->SetMarkerStyle(22);

      meEVT_ = m_dbe->bookInt("NZS Task Event Number");
      meEVT_->Fill(ievt_);    
      meTOTALEVT_ = m_dbe->bookInt("NZS Task Total Events Processed");
      meTOTALEVT_->Fill(tevt_);
    } // if (m_dbe)
  return;
}

void HcalNZSMonitor::processEvent(const FEDRawDataCollection& rawraw,
                                  const edm::TriggerResults & trigRes,
                                  int bxNum,
                                  const edm::TriggerNames & triggerNames)
{
  const unsigned int nTrig(triggerNames.size());
 
  vector<bool> trigAcc;
  for (unsigned int i=0; i<triggers_.size(); i++) trigAcc.push_back(false);
  
   for (unsigned int k=0; k<nTrig; k++)
     {
       for (unsigned int i=0; i<triggers_.size(); i++)
	 {
	   if (triggerNames.triggerName(k) == triggers_[i] && trigRes.accept(k)) trigAcc[i]=true;
	 }
     }
   bool andAcc=true;
   bool orAcc=false;
   for (unsigned int p=0; p<triggers_.size(); p++)
     {
       if (!trigAcc[p]) andAcc=false;
       if (trigAcc[p]) 
	 {
	   orAcc=true;
	   nAcc[p]++;
	 }
     }
   if (andAcc) 
     {
       nAndAcc++;
       meTrigFrac_->setBinContent(triggers_.size()+1,nAndAcc/(float)nAcc_Total);
     }
   if (orAcc)
     {
       nAcc_Total++;
       for (unsigned int l=0; l<triggers_.size(); l++)
	 {
	   meTrigFrac_->setBinContent(l+1,nAcc[l]/(float)nAcc_Total);
	 }
     }
   
   
     
  bool processevent=false;
  if (orAcc) processevent=true;

  if (!processevent) return;

  meBXtriggered_->Fill(bxNum+0.001,1);

  if(!m_dbe) 
    { 
      printf("HcalNZSMonitor::processEvent DQMStore not instantiated!!!\n");  
      return;
    }

  HcalBaseMonitor::processEvent();

  //calculate full HCAL data size:
  size_t hcalSize=0;
  bool hcalIsZS = false;
  for (unsigned int k=0; k<selFEDs_.size(); k++)
    {
      const FEDRawData & fedData = rawraw.FEDData(selFEDs_[k]);
      hcalSize+=fedData.size();

      meFEDsizesNZS_->Fill(selFEDs_[k]+0.001,fedData.size()/1024);

      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fedData.data());

      HcalHTRData htr;
      
      int nspigot =0; 
      for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
	{    
	  nspigot++;
	  
	  if (!dccHeader->getSpigotPresent(spigot)) continue;
	  
	  // Load the given decoder with the pointer and length from this spigot.
	  dccHeader->getSpigotData(spigot,htr, fedData.size()); 
	  
	  if(k != 20 && nspigot !=14 ) 
	    {      
	      if ( !htr.isUnsuppressed() ) hcalIsZS = true; 
	    }
	}
    }
  
  if (hcalIsZS) meIsUS_->Fill(0,1);
  else meIsUS_->Fill(1,1);

  meFEDsizeVsLumi_->Fill(lumiblock+0.001, hcalSize/1024);
  
  size_t fullSize=0;
  for (int j=0; j<FEDNumbering::MAXFEDID; ++j )
    {
      const FEDRawData & fedData = rawraw.FEDData(j);
      fullSize+=fedData.size();
    }

  meFullCMSdataSize_->Fill(lumiblock+0.001,fullSize/1024);
  
  // get Trigger FED-Id
  const FEDRawData& fedData = rawraw.FEDData(FEDNumbering::MINTriggerGTPFEDID) ;
  FEDHeader header(fedData.data()) ;
  
  /// Level-1 event number generated by the TTC system
  if (header.lvl1ID()%period_==0) meL1evtNumber_->Fill(1,1);
  else meL1evtNumber_->Fill(0,1);
 
  return;

}

void HcalNZSMonitor::endLuminosityBlock(void) {
  if (LBprocessed_==true) return;  // LB already processed
  UpdateMEs();
  LBprocessed_=true; 
  return;
}

void HcalNZSMonitor::UpdateMEs (void ) {}

