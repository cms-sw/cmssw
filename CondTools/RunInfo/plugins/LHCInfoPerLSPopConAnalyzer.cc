#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using std::make_pair;
using std::pair;

class LHCInfoPerLSPopConSourceHandler;

typedef popcon::PopConAnalyzer<LHCInfoPerLSPopConSourceHandler> LHCInfoPerLSPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(LHCInfoPerLSPopConAnalyzer);

namespace theLHCInfoPerLSImpl {
  bool comparePayloads(const LHCInfoPerLS& rhs, const LHCInfoPerLS& lhs) {
    if (rhs.fillNumber() != lhs.fillNumber() || rhs.runNumber() != lhs.runNumber() ||
        rhs.crossingAngleX() != lhs.crossingAngleX() || rhs.crossingAngleY() != lhs.crossingAngleY() ||
        rhs.betaStarX() != lhs.betaStarX() || rhs.betaStarY() != lhs.betaStarY()) {
      return false;
    }
    return true;
  }

  size_t transferPayloads(const std::vector<pair<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>>& buffer,
                          std::map<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>& iovsToTransfer,
                          std::shared_ptr<LHCInfoPerLS>& prevPayload,
                          const std::map<pair<cond::Time_t, unsigned int>, pair<cond::Time_t, unsigned int>>& lsIdMap,
                          cond::Time_t startStableBeamTime,
                          cond::Time_t endStableBeamTime) {
    int lsMissingInPPS = 0;
    int xAngleBothZero = 0, xAngleBothNonZero = 0, xAngleNegative = 0;
    int betaNegative = 0;
    size_t niovs = 0;
    std::stringstream condIovs;
    std::stringstream missingLsList;
    for (auto& iov : buffer) {
      bool add = false;
      auto payload = iov.second;
      cond::Time_t since = iov.first;
      if (iovsToTransfer.empty()) {
        add = true;
      } else {
        LHCInfoPerLS& lastAdded = *iovsToTransfer.rbegin()->second;
        if (!comparePayloads(lastAdded, *payload)) {
          add = true;
        }
      }
      auto id = make_pair(payload->runNumber(), payload->lumiSection());
      bool stableBeam = since >= startStableBeamTime && since <= endStableBeamTime;
      bool isMissing = lsIdMap.find(id) != lsIdMap.end() && id != lsIdMap.at(id);
      if (stableBeam && isMissing) {
        missingLsList << id.first << "_" << id.second << " ";
        lsMissingInPPS += isMissing;
      }
      if (add && !isMissing) {
        niovs++;
        if (stableBeam) {
          if (payload->crossingAngleX() == 0 && payload->crossingAngleY() == 0)
            xAngleBothZero++;
          if (payload->crossingAngleX() != 0 && payload->crossingAngleY() != 0)
            xAngleBothNonZero++;
          if (payload->crossingAngleX() < 0 || payload->crossingAngleY() < 0)
            xAngleNegative++;
          if (payload->betaStarX() < 0 || payload->betaStarY() < 0)
            betaNegative++;
        }

        condIovs << since << " ";
        iovsToTransfer.insert(make_pair(since, payload));
        prevPayload = iov.second;
      }
    }
    unsigned short fillNumber = (!buffer.empty()) ? buffer.front().second->fillNumber() : 0;
    if (lsMissingInPPS > 0) {
      edm::LogWarning("transferPayloads")
          << "Number of stable beam LS in OMS without corresponding record in PPS DB for fill " << fillNumber << ": "
          << lsMissingInPPS;
      edm::LogWarning("transferPayloads")
          << "Stable beam LS in OMS without corresponding record in PPS DB (run_LS):  " << missingLsList.str();
    }
    if (xAngleBothZero > 0) {
      edm::LogWarning("transferPayloads")
          << "Number of payloads written with crossingAngle == 0 for both X and Y for fill " << fillNumber << ": "
          << xAngleBothZero;
    }
    if (xAngleBothNonZero > 0) {
      edm::LogWarning("transferPayloads")
          << "Number of payloads written with crossingAngle != 0 for both X and Y for fill " << fillNumber << ": "
          << xAngleBothNonZero;
    }
    if (xAngleNegative > 0) {
      edm::LogWarning("transferPayloads")
          << "Number of payloads written with negative crossingAngle for fill " << fillNumber << ": " << xAngleNegative;
    }
    if (betaNegative > 0) {
      edm::LogWarning("transferPayloads")
          << "Number of payloads written with negative betaSta for fill " << fillNumber << ": " << betaNegative;
    }

    edm::LogInfo("transferPayloads") << "TRANSFERED COND IOVS: " << condIovs.str();
    return niovs;
  }

}  // namespace theLHCInfoPerLSImpl
class LHCInfoPerLSPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfoPerLS> {
public:
  LHCInfoPerLSPopConSourceHandler(edm::ParameterSet const& pset)
      : m_debug(pset.getUntrackedParameter<bool>("debug", false)),
        m_startTime(),
        m_endTime(),
        m_samplingInterval((unsigned int)pset.getUntrackedParameter<unsigned int>("samplingInterval", 300)),
        m_endFillMode(pset.getUntrackedParameter<bool>("endFill", true)),
        m_name(pset.getUntrackedParameter<std::string>("name", "LHCInfoPerLSPopConSourceHandler")),
        m_connectionString(pset.getUntrackedParameter<std::string>("connectionString", "")),
        m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema", "")),
        m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath", "")),
        m_omsBaseUrl(pset.getUntrackedParameter<std::string>("omsBaseUrl", "")),
        m_fillPayload(),
        m_prevPayload(),
        m_tmpBuffer() {
    if (!pset.getUntrackedParameter<std::string>("startTime").empty()) {
      m_startTime = boost::posix_time::time_from_string(pset.getUntrackedParameter<std::string>("startTime"));
    }
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    m_endTime = now;
    if (!pset.getUntrackedParameter<std::string>("endTime").empty()) {
      m_endTime = boost::posix_time::time_from_string(pset.getUntrackedParameter<std::string>("endTime"));
      if (m_endTime > now)
        m_endTime = now;
    }
  }
  //L1: try with different m_dipSchema
  //L2: try with different m_name
  ~LHCInfoPerLSPopConSourceHandler() override = default;
  void getNewObjects() override {
    //if a new tag is created, transfer fake fill from 1 to the first fill for the first time
    if (tagInfo().size == 0) {
      edm::LogInfo(m_name) << "New tag " << tagInfo().name << "; from " << m_name << "::getNewObjects";
    } else {
      //check what is already inside the database
      edm::LogInfo(m_name) << "got info for tag " << tagInfo().name << ": size " << tagInfo().size
                           << ", last object valid since " << tagInfo().lastInterval.since << " ( "
                           << boost::posix_time::to_iso_extended_string(
                                  cond::time::to_boost(tagInfo().lastInterval.since))
                           << " ); from " << m_name << "::getNewObjects";
    }

    cond::Time_t lastSince = tagInfo().lastInterval.since;
    if (tagInfo().isEmpty()) {
      // for a new or empty tag, an empty payload should be added on top with since=1
      addEmptyPayload(1);
      lastSince = 1;
    } else {
      edm::LogInfo(m_name) << "The last Iov in tag " << tagInfo().name << " valid since " << lastSince << "from "
                           << m_name << "::getNewObjects";
    }

    boost::posix_time::ptime executionTime = boost::posix_time::second_clock::local_time();
    cond::Time_t targetSince = 0;
    cond::Time_t executionTimeIov = cond::time::from_boost(executionTime);
    if (!m_startTime.is_not_a_date_time()) {
      targetSince = cond::time::from_boost(m_startTime);
    }
    if (lastSince > targetSince)
      targetSince = lastSince;

    edm::LogInfo(m_name) << "Starting sampling at "
                         << boost::posix_time::to_simple_string(cond::time::to_boost(targetSince));

    //retrieve the data from the relational database source
    cond::persistency::ConnectionPool connection;
    //configure the connection
    if (m_debug) {
      connection.setMessageVerbosity(coral::Debug);
    } else {
      connection.setMessageVerbosity(coral::Error);
    }
    connection.setAuthenticationPath(m_authpath);
    connection.configure();
    //create the sessions
    cond::persistency::Session session = connection.createSession(m_connectionString, false);
    // fetch last payload when available
    if (!tagInfo().lastInterval.payloadId.empty()) {
      cond::persistency::Session session3 = dbSession();
      session3.transaction().start(true);
      m_prevPayload = session3.fetchPayload<LHCInfoPerLS>(tagInfo().lastInterval.payloadId);
      session3.transaction().commit();

      // find startFillTime and endFillTime of the most recent fill already saved in the tag
      if (m_prevPayload->fillNumber() != 0) {
        cond::OMSService oms;
        oms.connect(m_omsBaseUrl);
        auto query = oms.query("fills");
        query->addOutputVar("end_time");
        query->addOutputVar("start_time");
        query->filterEQ("fill_number", m_prevPayload->fillNumber());
        bool foundFill = query->execute();
        if (foundFill) {
          auto result = query->result();

          if (!result.empty()) {
            std::string endTimeStr = (*result.begin()).get<std::string>("end_time");
            m_prevEndFillTime =
                (endTimeStr == "null")
                    ? 0
                    : cond::time::from_boost((*result.begin()).get<boost::posix_time::ptime>("end_time"));
            auto startFillTime = (*result.begin()).get<boost::posix_time::ptime>("start_time");
            m_prevStartFillTime = cond::time::from_boost(startFillTime);
          } else {
            foundFill = false;
          }
        }
        if (!foundFill) {
          edm::LogError(m_name) << "Could not find end time of fill #" << m_prevPayload->fillNumber();
        }
      } else {
        m_prevEndFillTime = 0;
        m_prevStartFillTime = 0;
      }
    }

    while (true) {
      if (targetSince >= executionTimeIov) {
        edm::LogInfo(m_name) << "Sampling ended at the time "
                             << boost::posix_time::to_simple_string(cond::time::to_boost(executionTimeIov));
        break;
      }
      boost::posix_time::ptime targetTime = cond::time::to_boost(targetSince);
      boost::posix_time::ptime startSampleTime;
      boost::posix_time::ptime endSampleTime;

      cond::OMSService oms;
      oms.connect(m_omsBaseUrl);
      auto query = oms.query("fills");

      if (!m_endFillMode and m_prevPayload->fillNumber() and m_prevEndFillTime == 0ULL) {
        // continue processing unfinished fill with some payloads already in the tag
        edm::LogInfo(m_name) << "Searching started fill #" << m_prevPayload->fillNumber();
        query->filterEQ("fill_number", m_prevPayload->fillNumber());
        bool foundFill = query->execute();
        if (foundFill)
          foundFill = makeFillPayload(m_fillPayload, query->result());
        if (!foundFill) {
          edm::LogError(m_name) << "Could not find fill #" << m_prevPayload->fillNumber();
          break;
        }
        startSampleTime = cond::time::to_boost(lastSince);
      } else {
        edm::LogInfo(m_name) << "Searching new fill after " << boost::posix_time::to_simple_string(targetTime);
        query->filterNotNull("start_stable_beam").filterNotNull("fill_number");
        if (targetTime > cond::time::to_boost(m_prevStartFillTime)) {
          query->filterGE("start_time", targetTime);
        } else {
          query->filterGT("start_time", targetTime);
        }

        query->filterLT("start_time", m_endTime);
        if (m_endFillMode)
          query->filterNotNull("end_time");
        bool foundFill = query->execute();
        if (foundFill)
          foundFill = makeFillPayload(m_fillPayload, query->result());
        if (!foundFill) {
          edm::LogInfo(m_name) << "No fill found - END of job.";
          break;
        }
        startSampleTime = cond::time::to_boost(m_startFillTime);
      }

      unsigned short lhcFill = m_fillPayload->fillNumber();
      bool ongoingFill = m_endFillTime == 0ULL;
      if (ongoingFill) {
        edm::LogInfo(m_name) << "Found ongoing fill " << lhcFill << " created at "
                             << cond::time::to_boost(m_startFillTime);
        endSampleTime = executionTime;
        targetSince = executionTimeIov;
      } else {
        edm::LogInfo(m_name) << "Found fill " << lhcFill << " created at " << cond::time::to_boost(m_startFillTime)
                             << " ending at " << cond::time::to_boost(m_endFillTime);
        endSampleTime = cond::time::to_boost(m_endFillTime);
        targetSince = m_endFillTime;
      }

      if (m_endFillMode || ongoingFill) {
        getLumiData(oms, lhcFill, startSampleTime, endSampleTime);

        if (!m_tmpBuffer.empty()) {
          boost::posix_time::ptime flumiStart = cond::time::to_boost(m_tmpBuffer.front().first);
          boost::posix_time::ptime flumiStop = cond::time::to_boost(m_tmpBuffer.back().first);
          edm::LogInfo(m_name) << "First buffered lumi starts at " << flumiStart << " last lumi starts at "
                               << flumiStop;
          session.transaction().start(true);
          getCTPPSData(session, startSampleTime, endSampleTime);
          session.transaction().commit();
        }
      }

      size_t niovs = theLHCInfoPerLSImpl::transferPayloads(
          m_tmpBuffer, m_iovs, m_prevPayload, m_lsIdMap, m_startStableBeamTime, m_endStableBeamTime);
      edm::LogInfo(m_name) << "Added " << niovs << " iovs within the Fill time";
      if (niovs) {
        m_prevEndFillTime = m_endFillTime;
        m_prevStartFillTime = m_startFillTime;
      }
      m_tmpBuffer.clear();
      m_lsIdMap.clear();
      if (m_prevPayload->fillNumber() and !ongoingFill)
        addEmptyPayload(m_endFillTime);
    }
  }
  std::string id() const override { return m_name; }

  static constexpr unsigned int kLumisectionsQueryLimit = 4000;

private:
  void addEmptyPayload(cond::Time_t iov) {
    bool add = false;
    if (m_iovs.empty()) {
      if (!m_lastPayloadEmpty)
        add = true;
    } else {
      auto lastAdded = m_iovs.rbegin()->second;
      if (lastAdded->fillNumber() != 0) {
        add = true;
      }
    }
    if (add) {
      auto newPayload = std::make_shared<LHCInfoPerLS>();
      m_iovs.insert(make_pair(iov, newPayload));
      m_prevPayload = newPayload;
      m_prevEndFillTime = 0;
      m_prevStartFillTime = 0;
      edm::LogInfo(m_name) << "Added empty payload with IOV" << iov << " ( "
                           << boost::posix_time::to_iso_extended_string(cond::time::to_boost(iov)) << " )";
    }
  }

  bool makeFillPayload(std::unique_ptr<LHCInfoPerLS>& targetPayload, const cond::OMSServiceResult& queryResult) {
    bool ret = false;
    if (!queryResult.empty()) {
      auto row = *queryResult.begin();
      auto currentFill = row.get<unsigned short>("fill_number");
      m_startFillTime = cond::time::from_boost(row.get<boost::posix_time::ptime>("start_time"));
      std::string endTimeStr = row.get<std::string>("end_time");
      m_endFillTime =
          (endTimeStr == "null") ? 0 : cond::time::from_boost(row.get<boost::posix_time::ptime>("end_time"));
      m_startStableBeamTime = cond::time::from_boost(row.get<boost::posix_time::ptime>("start_stable_beam"));
      m_endStableBeamTime = cond::time::from_boost(row.get<boost::posix_time::ptime>("end_stable_beam"));
      targetPayload = std::make_unique<LHCInfoPerLS>();
      targetPayload->setFillNumber(currentFill);
      ret = true;
    }
    return ret;
  }

  void addPayloadToBuffer(cond::OMSServiceResultRef& row) {
    auto lumiTime = row.get<boost::posix_time::ptime>("start_time");
    LHCInfoPerLS* thisLumiSectionInfo = new LHCInfoPerLS(*m_fillPayload);
    thisLumiSectionInfo->setLumiSection(std::stoul(row.get<std::string>("lumisection_number")));
    thisLumiSectionInfo->setRunNumber(std::stoull(row.get<std::string>("run_number")));
    m_lsIdMap[make_pair(thisLumiSectionInfo->runNumber(), thisLumiSectionInfo->lumiSection())] = make_pair(-1, -1);
    m_tmpBuffer.emplace_back(make_pair(cond::time::from_boost(lumiTime), thisLumiSectionInfo));
  }

  size_t bufferAllLS(const cond::OMSServiceResult& queryResult) {
    for (auto r : queryResult) {
      addPayloadToBuffer(r);
    }
    return queryResult.size();
  }

  size_t bufferFirstStableBeamLS(const cond::OMSServiceResult& queryResult) {
    for (auto r : queryResult) {
      if (r.get<std::string>("beams_stable") == "true") {
        addPayloadToBuffer(r);
        edm::LogInfo(m_name) << "Buffered first lumisection of stable beam: LS: "
                             << r.get<std::string>("lumisection_number")
                             << " run: " << r.get<std::string>("run_number");
        return 1;
      }
    }
    return 0;
  }

  size_t getLumiData(const cond::OMSService& oms,
                     unsigned short fillId,
                     const boost::posix_time::ptime& beginFillTime,
                     const boost::posix_time::ptime& endFillTime) {
    auto query = oms.query("lumisections");
    query->addOutputVars({"start_time", "run_number", "beams_stable", "lumisection_number"});
    query->filterEQ("fill_number", fillId);
    query->filterGT("start_time", beginFillTime).filterLT("start_time", endFillTime);
    query->limit(kLumisectionsQueryLimit);
    size_t nlumi = 0;
    if (query->execute()) {
      auto queryResult = query->result();
      if (m_endFillMode) {
        nlumi = bufferAllLS(queryResult);
      } else if (!queryResult.empty()) {
        auto newestPayload = queryResult.back();
        if (newestPayload.get<std::string>("beams_stable") == "true") {
          addPayloadToBuffer(newestPayload);
          nlumi = 1;
          edm::LogInfo(m_name) << "Buffered most recent lumisection:"
                               << " LS: " << newestPayload.get<std::string>("lumisection_number")
                               << " run: " << newestPayload.get<std::string>("run_number");
        }
      }
      edm::LogInfo(m_name) << "Found " << queryResult.size() << " lumisections during the fill " << fillId;
    } else {
      edm::LogInfo(m_name) << "OMS query for lumisections of fill " << fillId << "failed, status:" << query->status();
    }
    return nlumi;
  }

  bool getCTPPSData(cond::persistency::Session& session,
                    const boost::posix_time::ptime& beginFillTime,
                    const boost::posix_time::ptime& endFillTime) {
    //run the fifth query against the CTPPS schema
    //Initializing the CMS_CTP_CTPPS_COND schema.
    coral::ISchema& CTPPS = session.coralSession().schema("CMS_PPS_SPECT_COND");
    //execute query for CTPPS Data
    std::unique_ptr<coral::IQuery> CTPPSDataQuery(CTPPS.newQuery());
    //FROM clause
    CTPPSDataQuery->addToTableList(std::string("PPS_LHC_MACHINE_PARAMS"));
    //SELECT clause
    CTPPSDataQuery->addToOutputList(std::string("DIP_UPDATE_TIME"));
    CTPPSDataQuery->addToOutputList(std::string("LUMI_SECTION"));
    CTPPSDataQuery->addToOutputList(std::string("RUN_NUMBER"));
    CTPPSDataQuery->addToOutputList(std::string("FILL_NUMBER"));
    CTPPSDataQuery->addToOutputList(std::string("XING_ANGLE_P5_X_URAD"));
    CTPPSDataQuery->addToOutputList(std::string("XING_ANGLE_P5_Y_URAD"));
    CTPPSDataQuery->addToOutputList(std::string("BETA_STAR_P5_X_M"));
    CTPPSDataQuery->addToOutputList(std::string("BETA_STAR_P5_Y_M"));
    //WHERE CLAUSE
    coral::AttributeList CTPPSDataBindVariables;
    CTPPSDataBindVariables.extend<coral::TimeStamp>(std::string("beginFillTime"));
    CTPPSDataBindVariables.extend<coral::TimeStamp>(std::string("endFillTime"));
    CTPPSDataBindVariables[std::string("beginFillTime")].data<coral::TimeStamp>() = coral::TimeStamp(beginFillTime);
    CTPPSDataBindVariables[std::string("endFillTime")].data<coral::TimeStamp>() = coral::TimeStamp(endFillTime);
    std::string conditionStr = std::string("DIP_UPDATE_TIME>= :beginFillTime and DIP_UPDATE_TIME< :endFillTime");
    CTPPSDataQuery->setCondition(conditionStr, CTPPSDataBindVariables);
    //ORDER BY clause
    CTPPSDataQuery->addToOrderList(std::string("DIP_UPDATE_TIME"));
    //define query output
    coral::AttributeList CTPPSDataOutput;
    CTPPSDataOutput.extend<coral::TimeStamp>(std::string("DIP_UPDATE_TIME"));
    CTPPSDataOutput.extend<int>(std::string("LUMI_SECTION"));
    CTPPSDataOutput.extend<int>(std::string("RUN_NUMBER"));
    CTPPSDataOutput.extend<int>(std::string("FILL_NUMBER"));
    CTPPSDataOutput.extend<float>(std::string("XING_ANGLE_P5_X_URAD"));
    CTPPSDataOutput.extend<float>(std::string("XING_ANGLE_P5_Y_URAD"));
    CTPPSDataOutput.extend<float>(std::string("BETA_STAR_P5_X_M"));
    CTPPSDataOutput.extend<float>(std::string("BETA_STAR_P5_Y_M"));
    CTPPSDataQuery->defineOutput(CTPPSDataOutput);
    //execute the query
    coral::ICursor& CTPPSDataCursor = CTPPSDataQuery->execute();
    unsigned int lumiSection = 0;
    cond::Time_t runNumber = 0;
    int fillNumber = 0;
    float crossingAngleX = 0., betaStarX = 0.;
    float crossingAngleY = 0., betaStarY = 0.;

    bool ret = false;
    int wrongFillNumbers = 0;
    std::stringstream wrongFills;
    std::vector<pair<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>>::iterator current = m_tmpBuffer.begin();
    while (CTPPSDataCursor.next()) {
      if (m_debug) {
        std::ostringstream CTPPS;
        CTPPSDataCursor.currentRow().toOutputStream(CTPPS);
      }
      coral::Attribute const& dipTimeAttribute = CTPPSDataCursor.currentRow()[std::string("DIP_UPDATE_TIME")];
      if (!dipTimeAttribute.isNull()) {
        ret = true;
        coral::Attribute const& lumiSectionAttribute = CTPPSDataCursor.currentRow()[std::string("LUMI_SECTION")];
        if (!lumiSectionAttribute.isNull()) {
          lumiSection = lumiSectionAttribute.data<int>();
        }
        coral::Attribute const& runNumberAttribute = CTPPSDataCursor.currentRow()[std::string("RUN_NUMBER")];
        if (!runNumberAttribute.isNull()) {
          runNumber = runNumberAttribute.data<int>();
        }
        coral::Attribute const& fillNumberAttribute = CTPPSDataCursor.currentRow()[std::string("FILL_NUMBER")];
        if (!fillNumberAttribute.isNull()) {
          fillNumber = fillNumberAttribute.data<int>();
        }
        coral::Attribute const& crossingAngleXAttribute =
            CTPPSDataCursor.currentRow()[std::string("XING_ANGLE_P5_X_URAD")];
        if (!crossingAngleXAttribute.isNull()) {
          crossingAngleX = crossingAngleXAttribute.data<float>();
        }
        coral::Attribute const& crossingAngleYAttribute =
            CTPPSDataCursor.currentRow()[std::string("XING_ANGLE_P5_Y_URAD")];
        if (!crossingAngleYAttribute.isNull()) {
          crossingAngleY = crossingAngleYAttribute.data<float>();
        }
        coral::Attribute const& betaStarXAttribute = CTPPSDataCursor.currentRow()[std::string("BETA_STAR_P5_X_M")];
        if (!betaStarXAttribute.isNull()) {
          betaStarX = betaStarXAttribute.data<float>();
        }
        coral::Attribute const& betaStarYAttribute = CTPPSDataCursor.currentRow()[std::string("BETA_STAR_P5_Y_M")];
        if (!betaStarYAttribute.isNull()) {
          betaStarY = betaStarYAttribute.data<float>();
        }
        if (current != m_tmpBuffer.end() && current->second->fillNumber() != fillNumber) {
          wrongFills << "( " << runNumber << "_" << lumiSection << " fill: OMS: " << current->second->fillNumber()
                     << " PPSdb: " << fillNumber << " ) ";
          wrongFillNumbers++;
        }
        for (;
             current != m_tmpBuffer.end() && make_pair(current->second->runNumber(), current->second->lumiSection()) <=
                                                 make_pair(runNumber, lumiSection);
             current++) {
          LHCInfoPerLS& payload = *(current->second);
          payload.setCrossingAngleX(crossingAngleX);
          payload.setCrossingAngleY(crossingAngleY);
          payload.setBetaStarX(betaStarX);
          payload.setBetaStarY(betaStarY);
          payload.setLumiSection(lumiSection);
          payload.setRunNumber(runNumber);
          if (m_lsIdMap.find(make_pair(payload.runNumber(), payload.lumiSection())) != m_lsIdMap.end()) {
            m_lsIdMap[make_pair(payload.runNumber(), payload.lumiSection())] = make_pair(runNumber, lumiSection);
          }
        }
      }
    }
    if (wrongFillNumbers) {
      edm::LogWarning("getCTPPSData") << "Number of records from PPS DB with fillNumber different from OMS: "
                                      << wrongFillNumbers;
      edm::LogWarning("getCTPPSData") << "Records from PPS DB with fillNumber different from OMS: " << wrongFills.str();
    }
    return ret;
  }

private:
  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  // sampling interval in seconds
  unsigned int m_samplingInterval;
  bool m_endFillMode = true;
  std::string m_name;
  //for reading from relational database source
  std::string m_connectionString, m_ecalConnectionString;
  std::string m_dipSchema, m_authpath;
  std::string m_omsBaseUrl;
  std::unique_ptr<LHCInfoPerLS> m_fillPayload;
  std::shared_ptr<LHCInfoPerLS> m_prevPayload;
  cond::Time_t m_startFillTime;
  cond::Time_t m_endFillTime;
  cond::Time_t m_prevEndFillTime;
  cond::Time_t m_prevStartFillTime;
  cond::Time_t m_startStableBeamTime;
  cond::Time_t m_endStableBeamTime;
  std::vector<pair<cond::Time_t, std::shared_ptr<LHCInfoPerLS>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
  //mapping of lumisections IDs (pairs of runnumber an LS number) found in OMS to the IDs they've been assignd from PPS DB
  //value pair(-1, -1) means lumisection corresponding to the key exists in OMS but no lumisection was matched from PPS
  std::map<pair<cond::Time_t, unsigned int>, pair<cond::Time_t, unsigned int>> m_lsIdMap;
};
