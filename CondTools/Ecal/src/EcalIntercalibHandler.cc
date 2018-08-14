#include "CondTools/Ecal/interface/EcalIntercalibHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"

#include<iostream>

const Int_t kEBChannels = 61200, kEEChannels = 14648;

popcon::EcalIntercalibHandler::EcalIntercalibHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalIntercalibHandler")) {

  edm::LogInfo("EcalIntercalib Source handler constructor\n");
  m_firstRun = static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
  m_file_type = ps.getParameter<std::string>("type");           // xml/txt
  m_file_name = ps.getParameter<std::string>("fileName");
}

popcon::EcalIntercalibHandler::~EcalIntercalibHandler() {}

void popcon::EcalIntercalibHandler::getNewObjects() {
  //  std::cout << "------- Ecal - > getNewObjects\n";
  std::ostringstream ss; 
  ss<<"ECAL ";
	
  unsigned long long  irun;
  std::string file_= m_file_name;
  edm::LogInfo("going to open file ") << file_;
	    	    
  //      EcalCondHeader   header;
  EcalIntercalibConstants * payload = new EcalIntercalibConstants;
  if(m_file_type == "xml")
    readXML(file_, *payload);
  else
    readTXT(file_, *payload);
  irun = m_firstRun;
  Time_t snc= (Time_t) irun ;

  popcon::PopConSourceHandler<EcalIntercalibConstants>::m_to_transfer.push_back(std::make_pair(payload, snc));
}

void popcon::EcalIntercalibHandler::readXML(const std::string& file_,
					    EcalFloatCondObjectContainer& record){
  std::string dummyLine, bid;
  std::ifstream fxml;
  fxml.open(file_);
  if(!fxml.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << file_;
    exit (1);
  }
  // header
  for( int i=0; i< 6; i++) {
    getline(fxml, dummyLine);   // skip first lines
    //	std::cout << dummyLine << std::endl;
  }
  fxml >> bid;
  std::string stt = bid.substr(7,5);
  std::istringstream iEB(stt);
  int nEB;
  iEB >> nEB;
  if(nEB != kEBChannels) {
    edm::LogInfo("strange number of EB channels ") << nEB;
    exit(-1);
  }
  fxml >> bid;   // <item_version>0</item_version>
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    EBDetId myEBDetId = EBDetId::unhashIndex(iChannel);
    fxml >> bid;
    std::size_t found = bid.find("</");
    stt = bid.substr(6, found - 6);
    float val = std::stof(stt);
    record[myEBDetId] = val;
  }
  for( int i=0; i< 5; i++) {
    getline(fxml, dummyLine);   // skip first lines
    //	std::cout << dummyLine << std::endl;
  }
  fxml >> bid;
  stt = bid.substr(7,5);
  std::istringstream iEE(stt);
  int nEE;
  iEE >> nEE;
  if(nEE != kEEChannels) {
    edm::LogInfo("strange number of EE channels ") << nEE;
    exit(-1);
  }
  fxml >> bid;   // <item_version>0</item_version>
  // now endcaps
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    EEDetId myEEDetId = EEDetId::unhashIndex(iChannel);
    fxml >> bid;
    std::size_t found = bid.find("</");
    stt = bid.substr(6, found - 6);
    float val = std::stof(stt);
    record[myEEDetId] = val;
  }
}

void popcon::EcalIntercalibHandler::readTXT(const std::string& file_,
					    EcalFloatCondObjectContainer& record){
  std::ifstream ftxt;
  ftxt.open(file_);
  if(!ftxt.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << file_;
    exit (1);
  }
  int number_of_lines = 0, eta, phi, x, y, z;
  float val;
  std::string line;
  while (std::getline(ftxt, line)) {
    if(number_of_lines < kEBChannels) {                                    // barrel
      sscanf(line.c_str(), "%i %i %i %f", &eta, &phi, &z, &val);
      EBDetId ebdetid(eta, phi, EBDetId::ETAPHIMODE);
      record[ebdetid] = val;
    }
    else {                                                                 // endcaps
      sscanf(line.c_str(), "%i %i %i %f", &x, &y, &z, &val);
      EEDetId eedetid(x, y, z, EEDetId::XYMODE);
      record[eedetid] = val;
    }
    number_of_lines++;
  }
  edm::LogInfo("Number of lines in text file: ") << number_of_lines;
  int kChannels = kEBChannels + kEEChannels;
  if(number_of_lines != kChannels)
  edm::LogInfo("wrong number of channels!  Please check ");
}
