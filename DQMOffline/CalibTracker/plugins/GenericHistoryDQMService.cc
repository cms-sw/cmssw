#include "DQM/SiStripHistoricInfoClient/plugins/SiStripPopConHistoryDQMBase.h"

/**
  @author D. Giordano
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/
class GenericHistoryDQM : public SiStripPopConHistoryDQMBase {
public:
  explicit GenericHistoryDQM(const edm::ParameterSet&)
    : SiStripPopConHistoryDQMBase(iConfig)
    , m_detectorID{iConfig.getParameter<uint32_t>("DetectorId")}
  {}

  virtual ~GenericHistoryDQM();
private:
  //Methods to be specified by each subdet
  uint32_t returnDetComponent(const MonitorElement* ME);
  bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity );
  bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity );

  uint32_t m_detectorID;
};

uint32_t GenericHistoryDQM::returnDetComponent(const MonitorElement* ME)
{
  LogTrace("GenericHistoryDQM") <<  "[GenericHistoryDQM::returnDetComponent] returning value defined in the configuration Pset \"DetectorId\"";
  return m_detectorID;
}

/// Example on how to define an user function for the statistic extraction
bool GenericHistoryDQM::setDBLabelsForUser(const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity )
{
  if(quantity=="userExample_XMax"){
    userDBContent.push_back(keyName+std::string("@")+std::string("userExample_XMax"));
  }
  else if(quantity=="userExample_mean"){
      userDBContent.push_back(keyName+std::string("@")+std::string("userExample_mean"));
  }
  else{
    edm::LogError("DQMHistoryServiceBase") 
      << "Quantity " << quantity
      << " cannot be handled\nAllowed quantities are" 
      << "\n  'stat'   that includes: entries, mean, rms"
      << "\n  'landau' that includes: landauPeak, landauPeakErr, landauSFWHM, landauChi2NDF"
      << "\n  'gauss'  that includes: gaussMean, gaussSigma, gaussChi2NDF"
      << "\n or a specific user quantity that should be implemented in the user functions GenericHistoryDQM::setDBLabelsForUser"
      << std::endl;
    return false;
  }
  return true;
}

bool GenericHistoryDQM::setDBValuesForUser(const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity )
{
  if(quantity=="userExample_XMax"){
    values.push_back( me->getTH1F()->GetXaxis()->GetBinCenter(me->getTH1F()->GetMaximumBin()));
  }
  else if(quantity=="userExample_mean"){
    values.push_back( me->getMean() );
  }
  else{
    return false;
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
typedef popcon::PopConAnalyzer<GenericHistoryDQM> GenericDQMHistoryPopCon;
DEFINE_FWK_MODULE(GenericDQMHistoryPopCon);
