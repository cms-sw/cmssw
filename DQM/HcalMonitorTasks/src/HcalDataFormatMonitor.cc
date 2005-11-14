#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "TH1F.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {}

HcalDataFormatMonitor::~HcalDataFormatMonitor() {}

void HcalDataFormatMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/DCCMonitor");
    m_meDCC_ERRWD =  m_dbe->book1D("DCC Error Words","DCC Error Words",12,0,12);
    MonitorElementRootH1* me_ = (MonitorElementRootH1*)m_meDCC_ERRWD;
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(1,"Overflow Err");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(2,"Buffer Busy");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(3,"Empty Event");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(4,"Rejected L1A");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(5,"Latency Err");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(6,"Latency Warn");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(7,"Optical Data");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(8,"Clock Err");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(9,"Bunch Err");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(10,"??1");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(11,"??2");
    ((TH1F*)(me_->operator->()))->GetXaxis()->SetBinLabel(12,"??3");
  }

  m_readoutMapSource = ps.getParameter<std::string>("readoutMapSource");
  m_fedUnpackList = ps.getParameter<std::vector<int> >("FEDs");
  m_firstFED = ps.getParameter<int>("HcalFirstFED");
  const std::string filePrefix("file://");
  if (m_readoutMapSource.find(filePrefix)==0) {
    std::string theFile=m_readoutMapSource;
    theFile.erase(0,filePrefix.length());
    std::cout << "HcalDataFormatMonitor::setup  Reading HcalMapping from '" << theFile << "'\n";
    m_readoutMap=HcalMappingTextFileReader::readFromFile(theFile.c_str(),true); // maintain L2E for no real reason
  }
  std::cout << "HcalDataFormatMonitor::setup  Will unpack FEDs ";
  for (unsigned int i=0; i<m_fedUnpackList.size(); i++) 
    std::cout << m_fedUnpackList[i] << " ";
  std::cout << std::endl;
  return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw)
{

  if(!m_dbe) { printf("HcalDataFormatMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return;}

  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=m_fedUnpackList.begin(); i!=m_fedUnpackList.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    //   std::cout << "Processing FED " << *i << std::endl;
    // look only at the potential ones, to save time.
    if (m_readoutMap->subdetectorPresent(HcalBarrel,*i-m_firstFED) 
	|| m_readoutMap->subdetectorPresent(HcalEndcap,*i-m_firstFED)) 
      unpack(fed,0,0,10);
    if (m_readoutMap->subdetectorPresent(HcalOuter,*i-m_firstFED)) 
      unpack(fed,0,0,10);
    if (m_readoutMap->subdetectorPresent(HcalForward,*i-m_firstFED)) 
      unpack(fed,0,0,10);
  }
  //      unpacker_.unpack(fed,*m_readoutMap,hf, htp);

}

void HcalDataFormatMonitor::unpack(const FEDRawData& raw, int f_offset, int f_start, int f_end){

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  //  int dccid=dccHeader->getSourceId()-f_offset;  
  // check the summary status
  
  // walk through the HTR data...
  HcalHTRData htr;
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;    
    dccHeader->getSpigotData(spigot,htr);
    // check
    if (!htr.check()) {
      // TODO: log error!
      continue;
    }
    // calculate "real" number of presamples
    //    int nps=htr.getNPS()-f_start;

    //    printf("Spigot: %d, dccid: %d, nps; %d, errorword 0x%x\n",spigot,dccid,nps,htr.getErrorsWord());
    int err = htr.getErrorsWord();
    for(int i=0; i<12; i++)
      m_meDCC_ERRWD->Fill(i,err&(0x01<<i));

  }

  return;

}
