#include "DQMOffline/CalibTracker/plugins/SiStripDQMHistoryReader.h"

SiStripDQMHistoryHelper::~SiStripDQMHistoryHelper() {}

void SiStripDQMHistoryHelper::scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, HDQMSummary* summary, const std::string& keyName, const std::vector<std::string>& Quantities)
{
  //
  // -- Scan full root file and fill module numbers and histograms
  //
  //-----------------------------------------------------------------------------------------------

  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::scanTreeAndFillSummary] keyName " << keyName;

  std::stringstream ss;

  // Use boost regex for more flexibility
  boost::regex re;
  try {
    re.assign(keyName);
  } catch ( boost::regex_error& e ) {
    std::cout << "Error: " << keyName << " is not a valid regular expression: \""
              << e.what() << "\"" << std::endl;
    std::cout << "Skip search for matches" << std::endl;
    return;
  }
  for ( const MonitorElement* me : MEs ) {
    // Name including path
    std::string me_name;
    if ( m_useFullPath ) {
      me_name = me->getFullname();
    } else {
      me_name = me->getName();
      // If the line does not start with a "^" add it
      if( me_name.find("^") != 0 ) {
        me_name = "^" + me_name;
      }
    }
    // regex_search has grep-like behaviour
    if ( boost::regex_search(me_name, re) ) {

      HDQMSummary::InputVector values;
      std::vector<std::string> userDBContent;

      ss << "\nFound compatible ME " << me_name << " for key " << keyName << std::endl;

      for ( const std::string& quant : Quantities ) {
        if ( quant  == "landau" ) {
          setDBLabelsForLandau(keyName, userDBContent);
          setDBValuesForLandau(me, values);
        } else if ( quant  == "gauss" ) {
          setDBLabelsForGauss(keyName, userDBContent);
          setDBValuesForGauss(me, values);
        } else if ( quant  == "stat" ) {
          setDBLabelsForStat(keyName, userDBContent);
          setDBValuesForStat(me, values);
        } else {
          setDBLabelsForUser(keyName, userDBContent, quant);
          setDBValuesForUser(me, values, quant);
        }
      }
      uint32_t detid = returnDetComponent(me);

      ss << "detid " << detid << " \n";
      for ( size_t i = 0; i < values.size(); ++i )
        ss << "Quantity " << userDBContent[i] << " value " << values[i] << std::endl;

      summary->put(detid, values, userDBContent);
    }
  }
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::scanTreeAndFillSummary] " << ss.str();
}

bool SiStripDQMHistoryHelper::setDBLabelsForLandau(const std::string& keyName, std::vector<std::string>& userDBContent)
{
  userDBContent.push_back(keyName+fSep+std::string("landauPeak"));
  userDBContent.push_back(keyName+fSep+std::string("landauPeakErr"));
  userDBContent.push_back(keyName+fSep+std::string("landauSFWHM"));
  userDBContent.push_back(keyName+fSep+std::string("landauChi2NDF"));
  return true;
}

bool SiStripDQMHistoryHelper::setDBLabelsForGauss(const std::string& keyName, std::vector<std::string>& userDBContent)
{
  userDBContent.push_back(keyName+fSep+std::string("gaussMean"));
  userDBContent.push_back(keyName+fSep+std::string("gaussSigma"));
  userDBContent.push_back(keyName+fSep+std::string("gaussChi2NDF"));
  return true;
}

bool SiStripDQMHistoryHelper::setDBLabelsForStat(const std::string& keyName, std::vector<std::string>& userDBContent)
{
  userDBContent.push_back(keyName+fSep+std::string("entries"));
  userDBContent.push_back(keyName+fSep+std::string("mean"));
  userDBContent.push_back(keyName+fSep+std::string("rms"));
  return true;
}

bool SiStripDQMHistoryHelper::setDBValuesForLandau(const MonitorElement* me, HDQMSummary::InputVector& values)
{
  m_fitME.doLanGaussFit(me);
  values.push_back( m_fitME.getLanGaussPar("mpv")    );
  values.push_back( m_fitME.getLanGaussParErr("mpv") );
  values.push_back( m_fitME.getLanGaussConv("fwhm")  );
  if (m_fitME.getFitnDof()!=0 ) values.push_back( m_fitME.getFitChi()/m_fitME.getFitnDof() );
  else                         values.push_back(-99.);
  return true;
}

bool SiStripDQMHistoryHelper::setDBValuesForGauss(const MonitorElement* me, HDQMSummary::InputVector& values)
{
  m_fitME.doGaussFit(me);
  values.push_back( m_fitME.getGaussPar("mean")  );
  values.push_back( m_fitME.getGaussPar("sigma") );
  if (m_fitME.getFitnDof()!=0 ) values.push_back( m_fitME.getFitChi()/m_fitME.getFitnDof() );
  else                         values.push_back(-99.);
  return true;
}

bool SiStripDQMHistoryHelper::setDBValuesForStat(const MonitorElement* me, HDQMSummary::InputVector& values)
{
  values.push_back( me->getEntries());
  values.push_back( me->getMean());
  values.push_back( me->getRMS());
  return true;
}
