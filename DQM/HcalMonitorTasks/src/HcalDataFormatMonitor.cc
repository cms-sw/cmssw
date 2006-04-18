#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {}

HcalDataFormatMonitor::~HcalDataFormatMonitor() {}

void HcalDataFormatMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

    
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/DataFormatMonitor");    

    string type = "HBHE Data Format Error Words";
    hbHists.m_DCC_ERRWD =  m_dbe->book1D(type.c_str(),type.c_str(),12,0,12);
    type = "HBHE Data Format Error Map";
    hbHists.m_ERR_MAP = m_dbe->book2D(type.c_str(),type.c_str(),59,-29.5,29.5,40,0,40);

    type = "HF Data Format Error Words";
    hfHists.m_DCC_ERRWD =  m_dbe->book1D(type.c_str(),type.c_str(),12,0,12);
    type = "HF Data Format Error Map";
    hfHists.m_ERR_MAP = m_dbe->book2D(type.c_str(),type.c_str(),59,-29.5,29.5,40,0,40);

    type = "HO Data Format Error Words";
    hoHists.m_DCC_ERRWD =  m_dbe->book1D(type.c_str(),type.c_str(),12,0,12);
    type = "HO Data Format Error Map";
    hoHists.m_ERR_MAP = m_dbe->book2D(type.c_str(),type.c_str(),59,-29.5,29.5,40,0,40);

  }
  m_fedUnpackList = ps.getParameter<vector<int> >("FEDs");
  m_firstFED = ps.getParameter<int>("HcalFirstFED");
  cout << "HcalDataFormatMonitor::setup  Will unpack FEDs ";
  for (unsigned int i=0; i<m_fedUnpackList.size(); i++) 
    cout << m_fedUnpackList[i] << " ";
  cout << endl;

  return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw, const HcalElectronicsMap& emap){

  if(!m_dbe) { printf("HcalDataFormatMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return;}
  
  for (vector<int>::const_iterator i=m_fedUnpackList.begin(); i!=m_fedUnpackList.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    //    cout << "Data format monitor: Processing FED " << *i << endl;
    // look only at the potential ones, to save time.
    unpack(fed,emap);
   }
}

void HcalDataFormatMonitor::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap){
  
  // get the DataFormat header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  // check the summary status
  if(!dccHeader) return;
  int dccid=dccHeader->getSourceId()-m_firstFED;  

  // walk through the HTR data...
  HcalHTRData htr;

  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;    
    dccHeader->getSpigotData(spigot,htr);
    // check
    if (!htr.check() || htr.isHistogramEvent()) continue;

    MonitorElement* tmpErr = 0; MonitorElement* tmpMap = 0;
    bool valid = false;
    for(int fc=0; fc<3 && !valid; fc++){
      for(int f=0; f<9 && !valid; f++){
	HcalElectronicsId eid(fc,f,spigot,dccid);
	eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
	DetId did=emap.lookup(eid);
	if (!did.null()) {
	  switch (((HcalSubdetector)did.subdetId())) {
	  case (HcalBarrel):
	  case (HcalEndcap): {
	    tmpErr = hbHists.m_DCC_ERRWD; tmpMap = hbHists.m_ERR_MAP; 
	    valid = true;
	  } break;
	  case (HcalOuter): {
	    tmpErr = hoHists.m_DCC_ERRWD; tmpMap = hoHists.m_ERR_MAP; 
	    valid = true;
	  } break;
	  case (HcalForward): {
	    tmpErr = hfHists.m_DCC_ERRWD; tmpMap = hfHists.m_ERR_MAP; 
	    valid = true;
	  } break;
	  default: break;
	  }
	}
      }
    }
    
    int err = htr.getErrorsWord();
    if(tmpErr!=NULL){
      for(int i=0; i<12; i++)
	tmpErr->Fill(i,err&(0x01<<i));    
    }
    if(err>0 && tmpMap!=NULL){
      tmpMap->Fill(htr.readoutVMECrateId(),htr.htrSlot());
    }    
  }
  
  return;
}
