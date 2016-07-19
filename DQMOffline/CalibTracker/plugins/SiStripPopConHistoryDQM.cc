#include "DQM/SiStripHistoricInfoClient/plugins/SiStripPopConHistoryDQMBase.h"

/**
 * @class SiStripPopConHistoryDQM
 * @author D. Giordano, A.-C. Le Bihan
 *
 * @EDAnalyzer to read DQM root file & insert summary informations to DB
 */
class SiStripPopConHistoryDQM : public SiStripPopConHistoryDQMBase
{
public:
  explicit SiStripPopConHistoryDQM(const edm::ParameterSet& pset)
    : SiStripPopConHistoryDQMBase(pset)
  {}

  virtual ~SiStripPopConHistoryDQM();
private:
  uint32_t returnDetComponent(const MonitorElement* ME);
  bool setDBLabelsForUser(const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity);
  bool setDBValuesForUser(const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity);
};

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

SiStripPopConHistoryDQM::~SiStripPopConHistoryDQM() {}

uint32_t SiStripPopConHistoryDQM::returnDetComponent(const MonitorElement* ME)
{
  LogTrace("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::returnDetComponent]";
  const std::string str{ME->getName()};
  const size_t __key_length__=7;
  const size_t __detid_length__=9;

  uint32_t layer=0,side=0;

  if ( str.find("__det__") != std::string::npos ) {
    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
  }
  //TIB
  else if ( str.find("TIB") != std::string::npos ) {
    if ( str.find("layer")!= std::string::npos )
      layer = atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return TIBDetId(layer,0,0,0,0,0).rawId();
  }
  //TOB
  else if ( str.find("TOB") != std::string::npos ) {
    if ( str.find("layer") != std::string::npos )
      layer = atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return TOBDetId(layer,0,0,0,0).rawId();
  }
  //TID
  else if ( str.find("TID") != std::string::npos ) {
    if ( str.find("side") != std::string::npos ) {
      side = atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      if ( str.find("wheel") != std::string::npos ) {
	layer = atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
    }
    return TIDDetId(side,layer,0,0,0,0).rawId();
  }
  //TEC
  else if ( str.find("TEC") != std::string::npos ) {
    if ( str.find("side") != std::string::npos ) {
      side = atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      if ( str.find("wheel") != std::string::npos ) {
	layer = atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
    }
    return TECDetId(side,layer,0,0,0,0,0).rawId();
  }
  else
    return SiStripDetId(DetId::Tracker,0).rawId(); //Full Tracker
}

//Example on how to define an user function for the statistic extraction
bool SiStripPopConHistoryDQM SiStripHistoryDQMService::setDBLabelsForUser(const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity)
{
  if (quantity == "user_2DYmean") {
    userDBContent.push_back(keyName+sep()+std::string("yMean"));
    userDBContent.push_back(keyName+sep()+std::string("yError"));
  } else {
    edm::LogError("SiStripHistoryDQMService") << "ERROR: quantity does not exist in SiStripHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }
  return true;
}

bool SiStripPopConHistoryDQM::setDBValuesForUser(const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity)
{
  if (quantity == "user_2DYmean") {
    TH2F* Hist = (TH2F*) me->getTH2F();
    values.push_back( Hist->GetMean(2) );
    values.push_back( Hist->GetRMS(2) );
  } else {
    edm::LogError("SiStripHistoryDQMService") << "ERROR: quantity does not exist in SiStripHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
typedef popcon::PopConAnalyzer<SiStripPopConHistoryDQM> SiStripDQMHistoryPopCon;
DEFINE_FWK_MODULE(SiStripDQMHistoryPopCon);
