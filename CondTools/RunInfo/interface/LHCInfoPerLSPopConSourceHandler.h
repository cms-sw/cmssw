#ifndef LHCInfoPerLSPopConSourceHandler_h
#define LHCInfoPerLSPopConSourceHandler_h

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class LHCInfoPerLSPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfoPerLS> {
public:
  LHCInfoPerLSPopConSourceHandler(edm::ParameterSet const& pset);
  ~LHCInfoPerLSPopConSourceHandler() override;
  void getNewObjects() override;
  std::string id() const override; 
  // std::string id() const override { return m_name; }

  static constexpr unsigned int kLumisectionsQueryLimit = 4000;

private:
  void addEmptyPayload(cond::Time_t iov);
  bool makeFillPayload(std::unique_ptr<LHCInfoPerLS>& targetPayload, const cond::OMSServiceResult& queryResult);
  void addPayloadToBuffer(cond::OMSServiceResultRef& row);
  size_t bufferAllLS(const cond::OMSServiceResult& queryResult);
  size_t getLumiData(const cond::OMSService& oms,
                     unsigned short fillId,
                     const boost::posix_time::ptime& beginFillTime,
                     const boost::posix_time::ptime& endFillTime);
  bool getCTPPSData(cond::persistency::Session& session,
                    const boost::posix_time::ptime& beginFillTime,
                    const boost::posix_time::ptime& endFillTime);

  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  bool m_endFillMode = true;
  std::string m_name;
  //for reading from relational database source
  std::string m_connectionString;
  std::string m_authpath;
  std::string m_omsBaseUrl;
  //makes duringFill interpret finished fills as ongoing fills and writing their last LS
  // (disabling the check if the last LS is in stable beams, although still only fills with stable beams are being processed)
  // also, it doesn't write empty payload at the end of a finished fill (because it's interpreted as ongoing)
  const bool m_debugLogic;
  std::unique_ptr<LHCInfoPerLS> m_fillPayload;
  std::shared_ptr<LHCInfoPerLS> m_prevPayload;
  cond::Time_t m_startFillTime;
  cond::Time_t m_endFillTime;
  cond::Time_t m_prevEndFillTime;
  cond::Time_t m_prevStartFillTime;
  cond::Time_t m_startStableBeamTime;
  cond::Time_t m_endStableBeamTime;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
  //mapping of lumisections IDs (pairs of runnumber an LS number) found in OMS to the IDs they've been assignd from PPS DB
  //value pair(-1, -1) means lumisection corresponding to the key exists in OMS but no lumisection was matched from PPS
  std::map<std::pair<cond::Time_t, unsigned int>, std::pair<cond::Time_t, unsigned int>> m_lsIdMap;
};

#endif