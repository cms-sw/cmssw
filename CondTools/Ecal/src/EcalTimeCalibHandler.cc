#include "CondTools/Ecal/interface/EcalTimeCalibHandler.h"

const Int_t kEBChannels = 61200, kEEChannels = 14648;

popcon::EcalTimeCalibHandler::EcalTimeCalibHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTimeCalibHandler")) {
  edm::LogInfo("EcalTimeCalib Source handler constructor\n");
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_file_type = ps.getParameter<std::string>("type");  // xml/txt
  m_file_name = ps.getParameter<std::string>("fileName");
}

popcon::EcalTimeCalibHandler::~EcalTimeCalibHandler() {}

void popcon::EcalTimeCalibHandler::getNewObjects() {
  std::ostringstream ss;
  ss << "ECAL ";

  std::string file_ = m_file_name;
  edm::LogInfo("going to open file ") << file_;

  //      EcalCondHeader   header;
  EcalTimeCalibConstants* payload = new EcalTimeCalibConstants;
  if (m_file_type == "xml")
    readXML(file_, *payload);
  else
    readTXT(file_, *payload);
  Time_t snc = (Time_t)m_firstRun;

  popcon::PopConSourceHandler<EcalTimeCalibConstants>::m_to_transfer.push_back(std::make_pair(payload, snc));
}

void popcon::EcalTimeCalibHandler::readXML(const std::string& file_, EcalFloatCondObjectContainer& record) {
  std::string dummyLine, bid;
  std::ifstream fxml;
  fxml.open(file_);
  if (!fxml.is_open()) {
    throw cms::Exception("ERROR : cannot open file ")
      << file_;
  }
  // header
  for (int i = 0; i < 6; i++) {
    getline(fxml, dummyLine);  // skip first lines
    //	std::cout << dummyLine << std::endl;
  }
  fxml >> bid;
  std::string stt = bid.substr(7, 5);
  std::istringstream iEB(stt);
  int nEB;
  iEB >> nEB;
  if (nEB != kEBChannels) {
    throw cms::Exception("Strange number of EB channels ") << nEB;
  }
  fxml >> bid;  // <item_version>0</item_version>
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    EBDetId myEBDetId = EBDetId::unhashIndex(iChannel);
    fxml >> bid;
    std::size_t found = bid.find("</");
    stt = bid.substr(6, found - 6);
    float val = std::stof(stt);
    record[myEBDetId] = val;
  }
  for (int i = 0; i < 5; i++) {
    getline(fxml, dummyLine);  // skip first lines
  }
  fxml >> bid;
  stt = bid.substr(7, 5);
  std::istringstream iEE(stt);
  int nEE;
  iEE >> nEE;
  if (nEE != kEEChannels) {
    throw cms::Exception("Strange number of EE channels ") << nEE;
  }
  fxml >> bid;  // <item_version>0</item_version>
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

void popcon::EcalTimeCalibHandler::readTXT(const std::string& file_, EcalFloatCondObjectContainer& record) {
  std::ifstream ftxt;
  ftxt.open(file_);
  if (!ftxt.is_open()) {
    throw cms::Exception("ERROR : cannot open file ")
      << file_;
  }
  int number_of_lines = 0, eta, phi, x, y, z;
  float val;
  std::string line;
  while (std::getline(ftxt, line)) {
    if (number_of_lines < kEBChannels) {  // barrel
      sscanf(line.c_str(), "%i %i %i %f", &eta, &phi, &z, &val);
      EBDetId ebdetid(eta, phi, EBDetId::ETAPHIMODE);
      record[ebdetid] = val;
    } else {  // endcaps
      sscanf(line.c_str(), "%i %i %i %f", &x, &y, &z, &val);
      EEDetId eedetid(x, y, z, EEDetId::XYMODE);
      record[eedetid] = val;
    }
    number_of_lines++;
  }
  edm::LogInfo("Number of lines in text file: ") << number_of_lines;
  int kChannels = kEBChannels + kEEChannels;
  if (number_of_lines != kChannels)
    throw cms::Exception("Wrong number of channels!  Please check ");
}
