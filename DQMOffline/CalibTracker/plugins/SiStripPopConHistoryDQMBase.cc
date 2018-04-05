#include "DQMOffline/CalibTracker/plugins/SiStripPopConHistoryDQMBase.h"

SiStripPopConHistoryDQMBase::SiStripPopConHistoryDQMBase(const edm::ParameterSet& iConfig)
  : SiStripDQMPopConSourceHandler<HDQMSummary>(iConfig)
  , SiStripDQMHistoryHelper(iConfig)
  , MEDir_{iConfig.getUntrackedParameter<std::string>("ME_DIR", "DQMData")}
  , histoList_{iConfig.getParameter<VParameters>("histoList")}
{
  edm::LogInfo("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::SiStripHistoryDQMService]";
}

SiStripPopConHistoryDQMBase::~SiStripPopConHistoryDQMBase()
{
  edm::LogInfo("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::~SiStripHistoryDQMService]";
}

bool SiStripPopConHistoryDQMBase::checkForCompatibility(const std::string& otherMetaData)
{
  if ( otherMetaData.empty() )
    return true;

  uint32_t previousRun=atoi(otherMetaData.substr(otherMetaData.find("Run ")+4).c_str());

  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::checkForCompatibility] extracted string " << previousRun ;
  return previousRun < getRunNumber();
}

void SiStripPopConHistoryDQMBase::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter& getter)
{
  m_obj = HDQMSummary();
  m_obj.setRunNr(getRunNumber());

  // DISCOVER SET OF HISTOGRAMS & QUANTITIES TO BE UPLOADED
  std::vector<std::string> userDBContent;
  for ( const auto& histoParams : histoList_ ) {
    const std::string keyName{histoParams.getUntrackedParameter<std::string>("keyName")};
    for ( const auto& quant : histoParams.getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract") ) {
      if  ( quant == "landau" )
	setDBLabelsForLandau(keyName, userDBContent);
      else if  ( quant == "gauss" )
	setDBLabelsForGauss(keyName, userDBContent);
      else if  ( quant == "stat" )
	setDBLabelsForStat(keyName, userDBContent);
      else
	setDBLabelsForUser(keyName, userDBContent, quant);
    }
  }
  m_obj.setUserDBContent(userDBContent);

  std::stringstream ss;
  ss << "[DQMHistoryServiceBase::scanTreeAndFillSummary] QUANTITIES TO BE INSERTED IN DB :" << std::endl;
  for ( const std::string& iCont : m_obj.getUserDBContent() ) {
    ss << iCont<< std::endl;
  }
  edm::LogInfo("HDQMSummary") << ss.str();

  // OPEN DQM FILE
  const std::vector<MonitorElement*> MEs = getter.getAllContents(MEDir_);

  // FILL SUMMARY
  edm::LogInfo("HDQMSummary") << "\nSTARTING TO FILL OBJECT ";
  for ( const auto& histoParams : histoList_ ) {
    const std::string keyName{histoParams.getUntrackedParameter<std::string>("keyName")};
    scanTreeAndFillSummary(MEs, &m_obj, keyName, histoParams.getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"));
  }
}

HDQMSummary* SiStripPopConHistoryDQMBase::getObj() const
{
  return new HDQMSummary(m_obj);
}
