#ifndef DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H
#define DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H

#include <memory>
#include "" // TODO include quite a lot

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
  uint32_t returnDetComponent(const MonitorElement* ME) = 0;
  std::string sep() const { return m_sep; }

  virtual void scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, HDQMSummary* summary, const std::string& histoName, const std::vector<std::string>& Quantities);

  virtual bool setDBLabelsForLandau(const std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForGauss (const std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForStat  (const std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForUser  (const std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity ) { return setDBLabelsForUser(keyName, userDBContent); }
  virtual bool setDBLabelsForUser  (const std::string& keyName, std::vector<std::string>& userDBContent) { return false; }

  virtual bool setDBValuesForLandau(const MonitorElement* me, HDQMSummary::InputVector& values);
  virtual bool setDBValuesForGauss (const MonitorElement* me, HDQMSummary::InputVector& values);
  virtual bool setDBValuesForStat  (const MonitorElement* me, HDQMSummary::InputVector& values);
  virtual bool setDBValuesForUser  (const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity ) { return setDBValuesForUser(me, values); }
  virtual bool setDBValuesForUser  (const MonitorElement* me, HDQMSummary::InputVector& values) { return false; }

private:
  bool m_useFullPath;
  std::string m_sep;
  HDQMfitUtilities m_fitME;
};

#endif // DQMOffline_CalibTracker_SiStripDQMHistoryHelper_H
