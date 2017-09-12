#ifndef DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H
#define DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"
class MonitorElement;

class SiStripDQMHistoryHelper
{
public:
  explicit SiStripDQMHistoryHelper(const edm::ParameterSet& pset)
    : m_useFullPath{pset.getUntrackedParameter<bool>("useFullPath", false)}
    , m_sep{"@"}
    , m_fitME{}
  {}
  virtual ~SiStripDQMHistoryHelper();

protected:
  virtual uint32_t returnDetComponent(const MonitorElement* ME) const = 0;
  std::string sep() const { return m_sep; }

  virtual void scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, HDQMSummary* summary, const std::string& histoName, const std::vector<std::string>& Quantities) const;

  virtual bool setDBLabelsForLandau(const std::string& keyName, std::vector<std::string>& userDBContent) const;
  virtual bool setDBLabelsForGauss (const std::string& keyName, std::vector<std::string>& userDBContent) const;
  virtual bool setDBLabelsForStat  (const std::string& keyName, std::vector<std::string>& userDBContent) const;
  virtual bool setDBLabelsForUser  (const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity ) const { return setDBLabelsForUser(keyName, userDBContent); }
  virtual bool setDBLabelsForUser  (const std::string& keyName, std::vector<std::string>& userDBContent) const { return false; }

  virtual bool setDBValuesForLandau(const MonitorElement* me, HDQMSummary::InputVector& values) const;
  virtual bool setDBValuesForGauss (const MonitorElement* me, HDQMSummary::InputVector& values) const;
  virtual bool setDBValuesForStat  (const MonitorElement* me, HDQMSummary::InputVector& values) const;
  virtual bool setDBValuesForUser  (const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity ) const { return setDBValuesForUser(me, values); }
  virtual bool setDBValuesForUser  (const MonitorElement* me, HDQMSummary::InputVector& values) const { return false; }

private:
  bool m_useFullPath;
  std::string m_sep;
  mutable HDQMfitUtilities m_fitME;
};

#endif // DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H
