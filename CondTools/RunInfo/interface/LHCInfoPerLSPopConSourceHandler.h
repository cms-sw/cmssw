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

private:
  void populateIovs();
  void filterInvalidPayloads();
  bool isPayloadValid(const LHCInfoPerLS& payload) const;
  void addEmptyPayload(cond::Time_t iov);
  void addDefaultPayload(cond::Time_t iov, unsigned short fill, const cond::OMSService& oms);
  void addDefaultPayload(cond::Time_t iov, unsigned short fill, int run, unsigned short lumi);
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
  // for reading from relational database source
  std::string m_connectionString;
  std::string m_authpath;
  std::string m_omsBaseUrl;
  // Allows for basic test of durigFill mode when there is no Stable Beams in LHC
  // makes duringFill interpret fills as ongoing fill and writing their last LS
  // (disabling the check if the last LS is in stable beams,
  // although still only fills with stable beams are being processed
  // also, still only up to one payload will be written)
  const bool m_debugLogic;
  // values for the default payload which is inserted after the last processed fill
  // has ended and there's no ongoing stable beam yet:
  float m_defaultCrossingAngleX;
  float m_defaultCrossingAngleY;
  float m_defaultBetaStarX;
  float m_defaultBetaStarY;
  float m_minBetaStar;       // meters
  float m_maxBetaStar;       // meters
  float m_minCrossingAngle;  // urad
  float m_maxCrossingAngle;  // urad

  std::unique_ptr<LHCInfoPerLS> m_fillPayload;
  std::shared_ptr<LHCInfoPerLS> m_prevPayload;
  cond::Time_t m_startFillTime;
  cond::Time_t m_endFillTime;
  cond::Time_t m_prevEndFillTime = 0;
  cond::Time_t m_prevStartFillTime;
  cond::Time_t m_startStableBeamTime;
  cond::Time_t m_endStableBeamTime;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
  // mapping of lumisections IDs (pairs of runnumber an LS number) found in OMS to
  // the IDs they've been assignd from PPS DB value pair(-1, -1) means lumisection
  // corresponding to the key exists in OMS but no lumisection was matched from PPS
  std::map<std::pair<cond::Time_t, unsigned int>, std::pair<cond::Time_t, unsigned int>> m_lsIdMap;
};

#endif