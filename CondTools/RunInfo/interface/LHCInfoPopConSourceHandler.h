#ifndef LHCINFOPOPCONSOURCEHANDLER_H
#define LHCINFOPOPCONSOURCEHANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace cond {
  class OMSService;
}

class LHCInfoPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfo> {
public:
  LHCInfoPopConSourceHandler(const edm::ParameterSet& pset);
  ~LHCInfoPopConSourceHandler() override;
  void getNewObjects() override;
  std::string id() const override;

  static constexpr unsigned int kLumisectionsQueryLimit = 4000;  // enough for fills not exceeding 25h

private:
  void addEmptyPayload(cond::Time_t iov);

  size_t getLumiData(const cond::OMSService& service,
                     unsigned short fillId,
                     const boost::posix_time::ptime& beginFillTime,
                     const boost::posix_time::ptime& endFillTime);
  void getDipData(const cond::OMSService& service,
                  const boost::posix_time::ptime& beginFillTime,
                  const boost::posix_time::ptime& endFillTime);
  bool getCTTPSData(cond::persistency::Session& session,
                    const boost::posix_time::ptime& beginFillTime,
                    const boost::posix_time::ptime& endFillTime);
  bool getEcalData(cond::persistency::Session& session,
                   const boost::posix_time::ptime& lowerTime,
                   const boost::posix_time::ptime& upperTime,
                   bool update);

private:
  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  // sampling interval in seconds
  unsigned int m_samplingInterval;
  bool m_endFill = true;
  std::string m_name;
  //for reading from relational database source
  std::string m_connectionString, m_ecalConnectionString;
  std::string m_dipSchema, m_authpath;
  std::string m_omsBaseUrl;
  std::unique_ptr<LHCInfo> m_fillPayload;
  std::shared_ptr<LHCInfo> m_prevPayload;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo> > > m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
};

#endif
