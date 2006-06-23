#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {}

HcalDataFormatMonitor::~HcalDataFormatMonitor() {}

void HcalDataFormatMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

    
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/DataFormatMonitor");
    m_DCC_ERRWD =  m_dbe->book1D("Data Format Error Words","Data Format Error Words",12,0,12);
    m_ERR_MAP = m_dbe->book2D("Data Format Error Map","Data Format Error Map",59,-29.5,29.5,40,0,40);
  }
  m_fedUnpackList = ps.getParameter<vector<int> >("FEDs");
  m_firstFED = ps.getParameter<int>("HcalFirstFED");
  cout << "HcalDataFormatMonitor::setup  Will unpack FEDs ";
  for (unsigned int i=0; i<m_fedUnpackList.size(); i++) 
    cout << m_fedUnpackList[i] << " ";
  cout << endl;

  return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw)
{

  if(!m_dbe) { printf("HcalDataFormatMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return;}

  for (vector<int>::const_iterator i=m_fedUnpackList.begin(); i!=m_fedUnpackList.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    //    cout << "Data format monitor: Processing FED " << *i << endl;
    // look only at the potential ones, to save time.
    unpack(fed);
   }
}

void HcalDataFormatMonitor::unpack(const FEDRawData& raw){
  
  // get the DataFormat header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  //  int dccid=dccHeader->getSourceId();  
  // check the summary status
  if(!dccHeader) return;
  // walk through the HTR data...
  HcalHTRData htr;
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;    
    dccHeader->getSpigotData(spigot,htr);
    // check
    if (!htr.check() || htr.isHistogramEvent()) continue;
    
    int err = htr.getErrorsWord();
    for(int i=0; i<12; i++)
      m_DCC_ERRWD->Fill(i,err&(0x01<<i));    

    if(err>0){
      m_ERR_MAP->Fill(htr.readoutVMECrateId(),htr.htrSlot());
    }

  }
  return;
}
