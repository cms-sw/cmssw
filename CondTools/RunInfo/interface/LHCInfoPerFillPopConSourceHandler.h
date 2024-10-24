#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class LHCInfoPerFillPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfoPerFill> {
public:
  LHCInfoPerFillPopConSourceHandler(edm::ParameterSet const& pset);
  ~LHCInfoPerFillPopConSourceHandler() override = default;

  void getNewObjects() override;
  std::string id() const override;

  static constexpr unsigned int kLumisectionsQueryLimit = 4000;

private:
  void addEmptyPayload(cond::Time_t iov);

  // Add payload to buffer and store corresponding lumiid IOV in m_timestampToLumiid map
  void addPayloadToBuffer(cond::OMSServiceResultRef& row);
  void convertBufferedIovsToLumiid(std::map<cond::Time_t, cond::Time_t> timestampToLumiid);

  size_t getLumiData(const cond::OMSService& oms,
                     unsigned short fillId,
                     const boost::posix_time::ptime& beginFillTime,
                     const boost::posix_time::ptime& endFillTime);

  void getDipData(const cond::OMSService& oms,
                  const boost::posix_time::ptime& beginFillTime,
                  const boost::posix_time::ptime& endFillTime);

  bool getCTPPSData(cond::persistency::Session& session,
                    const boost::posix_time::ptime& beginFillTime,
                    const boost::posix_time::ptime& endFillTime);

  bool getEcalData(cond::persistency::Session& session,
                   const boost::posix_time::ptime& lowerTime,
                   const boost::posix_time::ptime& upperTime);

private:
  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  bool m_endFillMode = true;
  std::string m_name;
  //for reading from relational database source
  std::string m_connectionString, m_ecalConnectionString;
  std::string m_authpath;
  std::string m_omsBaseUrl;
  std::unique_ptr<LHCInfoPerFill> m_fillPayload;
  std::shared_ptr<LHCInfoPerFill> m_prevPayload;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
  // to hold correspondance between timestamp-type IOVs and lumiid-type IOVs
  std::map<cond::Time_t, cond::Time_t> m_timestampToLumiid;
};
