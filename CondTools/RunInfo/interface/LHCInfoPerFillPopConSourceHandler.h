#pragma once

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

  bool isPayloadValid(const LHCInfoPerFill& payload) const;

protected:
  virtual std::unique_ptr<LHCInfoPerFill> findFillToProcess(cond::OMSService& oms,
                                                            const boost::posix_time::ptime& nextFillSearchTime,
                                                            bool inclusiveSearchTime);
  void addEmptyPayload(cond::Time_t iov);

  // Add payload to buffer and store corresponding lumiid IOV in m_timestampToLumiid map
  void addPayloadToBuffer(cond::OMSServiceResultRef& row);
  void convertBufferedIovsToLumiid(std::map<cond::Time_t, cond::Time_t> timestampToLumiid);

  /**
   * @return A tuple containing:
   *         - cond::OMSServiceResult: The result of the luminosity query.
   *         - bool: Indicates whether the query was successful.
   *         - std::unique_ptr<cond::OMSServiceQuery>: Owner object for the query.
   *           Query result resources are tied to its lifetime so it needs to be kept in the same scope.
   */
  virtual std::tuple<cond::OMSServiceResult, bool, std::unique_ptr<cond::OMSServiceQuery>> executeLumiQuery(
      const cond::OMSService& oms,
      unsigned short fillId,
      const boost::posix_time::ptime& beginFillTime,
      const boost::posix_time::ptime& endFillTime) const;

  virtual void getLumiData(const cond::OMSService& oms,
                           unsigned short fillId,
                           const boost::posix_time::ptime& beginFillTime,
                           const boost::posix_time::ptime& endFillTime);

  virtual void getDipData(const cond::OMSService& oms,
                          const boost::posix_time::ptime& beginFillTime,
                          const boost::posix_time::ptime& endFillTime);

  virtual bool getCTPPSData(cond::persistency::Session& session,
                            const boost::posix_time::ptime& beginFillTime,
                            const boost::posix_time::ptime& endFillTime);

  virtual bool getEcalData(cond::persistency::Session& session,
                           const boost::posix_time::ptime& lowerTime,
                           const boost::posix_time::ptime& upperTime);

  bool getCTPPSDataImpl(cond::persistency::Session& session,
                        const boost::posix_time::ptime& beginFillTime,
                        const boost::posix_time::ptime& endFillTime);

  bool getEcalDataImpl(cond::persistency::Session& session,
                       const boost::posix_time::ptime& lowerTime,
                       const boost::posix_time::ptime& upperTime);

protected:
  virtual std::tuple<cond::persistency::Session, cond::persistency::Session> createSubsystemDbSessions() const;
  virtual cond::Time_t getNextFillSearchTimestamp(cond::Time_t lastSince) const;
  virtual cond::Time_t handleIfNewTagAndGetLastSince();
  virtual void fetchLastPayload();
  virtual boost::posix_time::ptime getExecutionTime() const;
  void populateIovs();
  void handleInvalidPayloads();
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

  float m_minEnergy;  // [GeV], applicable in duringFill mode only
  float m_maxEnergy;  // [GeV], applicable in duringFill mode only
  bool m_throwOnInvalid = true;

  std::unique_ptr<LHCInfoPerFill> m_fillPayload;
  std::shared_ptr<LHCInfoPerFill> m_prevPayload;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
  // to hold correspondance between timestamp-type IOVs and lumiid-type IOVs
  std::map<cond::Time_t, cond::Time_t> m_timestampToLumiid;
};
