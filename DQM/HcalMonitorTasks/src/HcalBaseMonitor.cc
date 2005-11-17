#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
}

HcalBaseMonitor::~HcalBaseMonitor() {
}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  if(dbe != NULL) m_dbe = dbe;
  
  m_readoutMapSource = ps.getParameter<std::string>("readoutMapSource");
  const string filePrefix("file://");
  if (m_readoutMapSource.find(filePrefix)==0) {
    string theFile=m_readoutMapSource;
    theFile.erase(0,filePrefix.length());
    std::cout << "HcalDataFormatMonitor::setup  Reading HcalMapping from '" << theFile << "'\n";
    m_readoutMap = *HcalMappingTextFileReader::readFromFile(theFile.c_str(),true); // maintain L2E for no real reason
  }

}

void HcalBaseMonitor::done(){
  return;
}
