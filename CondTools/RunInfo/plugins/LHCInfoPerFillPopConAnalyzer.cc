#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondTools/RunInfo/interface/LumiSectionFilter.h"
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
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class LHCInfoPerFillPopConSourceHandler;

typedef popcon::PopConAnalyzer<LHCInfoPerFillPopConSourceHandler> LHCInfoPerFillPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(LHCInfoPerFillPopConAnalyzer);

namespace cond {
  namespace theLHCInfoPerFillPopConImpl {

    static const std::pair<const char*, LHCInfoPerFill::FillType> s_fillTypeMap[] = {
        std::make_pair("PROTONS", LHCInfoPerFill::PROTONS),
        std::make_pair("IONS", LHCInfoPerFill::IONS),
        std::make_pair("COSMICS", LHCInfoPerFill::COSMICS),
        std::make_pair("GAP", LHCInfoPerFill::GAP)};

    static const std::pair<const char*, LHCInfoPerFill::ParticleType> s_particleTypeMap[] = {
        std::make_pair("PROTON", LHCInfoPerFill::PROTON),
        std::make_pair("PB82", LHCInfoPerFill::PB82),
        std::make_pair("AR18", LHCInfoPerFill::AR18),
        std::make_pair("D", LHCInfoPerFill::D),
        std::make_pair("XE54", LHCInfoPerFill::XE54)};

    LHCInfoPerFill::FillType fillTypeFromString(const std::string& s_fill_type) {
      for (auto const& i : s_fillTypeMap)
        if (s_fill_type == i.first)
          return i.second;
      return LHCInfoPerFill::UNKNOWN;
    }

    LHCInfoPerFill::ParticleType particleTypeFromString(const std::string& s_particle_type) {
      for (auto const& i : s_particleTypeMap)
        if (s_particle_type == i.first)
          return i.second;
      return LHCInfoPerFill::NONE;
    }
  }  // namespace theLHCInfoPerFillPopConImpl

  namespace impl {

    template <>
    LHCInfoPerFill::FillType from_string(const std::string& attributeValue) {
      return from_string_impl<LHCInfoPerFill::FillType, &theLHCInfoPerFillPopConImpl::fillTypeFromString>(
          attributeValue, LHCInfoPerFill::UNKNOWN);
    }

    template <>
    LHCInfoPerFill::ParticleType from_string(const std::string& attributeValue) {
      return from_string_impl<LHCInfoPerFill::ParticleType, &theLHCInfoPerFillPopConImpl::particleTypeFromString>(
          attributeValue, LHCInfoPerFill::NONE);
    }

  }  // namespace impl
}  // namespace cond

namespace theLHCInfoPerFillImpl {

  bool makeFillPayload(std::unique_ptr<LHCInfoPerFill>& targetPayload, const cond::OMSServiceResult& queryResult) {
    bool ret = false;
    if (!queryResult.empty()) {
      auto row = *queryResult.begin();
      auto currentFill = row.get<unsigned short>("fill_number");
      auto bunches1 = row.get<unsigned short>("bunches_beam1");
      auto bunches2 = row.get<unsigned short>("bunches_beam2");
      auto collidingBunches = row.get<unsigned short>("bunches_colliding");
      auto targetBunches = row.get<unsigned short>("bunches_target");
      auto fillType = row.get<LHCInfoPerFill::FillType>("fill_type_runtime");
      auto particleType1 = row.get<LHCInfoPerFill::ParticleType>("fill_type_party1");
      auto particleType2 = row.get<LHCInfoPerFill::ParticleType>("fill_type_party2");
      auto intensityBeam1 = row.get<float>("intensity_beam1");
      auto intensityBeam2 = row.get<float>("intensity_beam2");
      auto energy = row.get<float>("energy");
      auto creationTime = row.get<boost::posix_time::ptime>("start_time");
      auto stableBeamStartTime = row.get<boost::posix_time::ptime>("start_stable_beam");
      std::string endTimeStr = row.get<std::string>("end_time");
      auto beamDumpTime =
          (endTimeStr == "null") ? 0 : cond::time::from_boost(row.get<boost::posix_time::ptime>("end_time"));
      auto injectionScheme = row.get<std::string>("injection_scheme");
      targetPayload = std::make_unique<LHCInfoPerFill>();
      targetPayload->setFillNumber(currentFill);
      targetPayload->setBunchesInBeam1(bunches1);
      targetPayload->setBunchesInBeam2(bunches2);
      targetPayload->setCollidingBunches(collidingBunches);
      targetPayload->setTargetBunches(targetBunches);
      targetPayload->setFillType(fillType);
      targetPayload->setParticleTypeForBeam1(particleType1);
      targetPayload->setParticleTypeForBeam2(particleType2);
      targetPayload->setIntensityForBeam1(intensityBeam1);
      targetPayload->setIntensityForBeam2(intensityBeam2);
      targetPayload->setEnergy(energy);
      targetPayload->setCreationTime(cond::time::from_boost(creationTime));
      targetPayload->setBeginTime(cond::time::from_boost(stableBeamStartTime));
      targetPayload->setEndTime(beamDumpTime);
      targetPayload->setInjectionScheme(injectionScheme);
      ret = true;
    }
    return ret;
  }
}  // namespace theLHCInfoPerFillImpl

namespace theLHCInfoPerFillImpl {
  static const std::map<std::string, int> vecMap = {
      {"Beam1/beamPhaseMean", 1}, {"Beam2/beamPhaseMean", 2}, {"Beam1/cavPhaseMean", 3}, {"Beam2/cavPhaseMean", 4}};
  void setElementData(cond::Time_t since,
                      const std::string& dipVal,
                      unsigned int elementNr,
                      float value,
                      LHCInfoPerFill& payload,
                      std::set<cond::Time_t>& initList) {
    if (initList.find(since) == initList.end()) {
      payload.beam1VC().resize(LHCInfoPerFill::bunchSlots, 0.);
      payload.beam2VC().resize(LHCInfoPerFill::bunchSlots, 0.);
      payload.beam1RF().resize(LHCInfoPerFill::bunchSlots, 0.);
      payload.beam2RF().resize(LHCInfoPerFill::bunchSlots, 0.);
      initList.insert(since);
    }
    // set the current values to all of the payloads of the lumi section samples after the current since
    if (elementNr < LHCInfoPerFill::bunchSlots) {
      switch (vecMap.at(dipVal)) {
        case 1:
          payload.beam1VC()[elementNr] = value;
          break;
        case 2:
          payload.beam2VC()[elementNr] = value;
          break;
        case 3:
          payload.beam1RF()[elementNr] = value;
          break;
        case 4:
          payload.beam2RF()[elementNr] = value;
          break;
        default:
          break;
      }
    }
  }
}  // namespace theLHCInfoPerFillImpl

namespace theLHCInfoPerFillImpl {
  bool comparePayloads(const LHCInfoPerFill& rhs, const LHCInfoPerFill& lhs) {
    if (rhs.fillNumber() != lhs.fillNumber() || rhs.delivLumi() != lhs.delivLumi() || rhs.recLumi() != lhs.recLumi() ||
        rhs.instLumi() != lhs.instLumi() || rhs.instLumiError() != lhs.instLumiError() ||
        rhs.lhcState() != lhs.lhcState() || rhs.lhcComment() != lhs.lhcComment() ||
        rhs.ctppsStatus() != lhs.ctppsStatus()) {
      return false;
    }
    return true;
  }

  size_t transferPayloads(const std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>>& buffer,
                          std::map<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>& iovsToTransfer,
                          std::shared_ptr<LHCInfoPerFill>& prevPayload) {
    size_t niovs = 0;
    std::stringstream condIovs;
    std::stringstream formattedIovs;
    for (auto& iov : buffer) {
      bool add = false;
      auto payload = iov.second;
      cond::Time_t since = iov.first;
      if (iovsToTransfer.empty()) {
        add = true;
      } else {
        LHCInfoPerFill& lastAdded = *iovsToTransfer.rbegin()->second;
        if (!comparePayloads(lastAdded, *payload)) {
          add = true;
        }
      }
      if (add) {
        niovs++;
        condIovs << since << " ";
        formattedIovs << boost::posix_time::to_iso_extended_string(cond::time::to_boost(since)) << " ";
        iovsToTransfer.insert(std::make_pair(since, payload));
        prevPayload = iov.second;
      }
    }
    edm::LogInfo("transferPayloads") << "TRANSFERED IOVS: " << condIovs.str();
    edm::LogInfo("transferPayloads") << "FORMATTED TRANSFERED IOVS: " << formattedIovs.str();
    return niovs;
  }

}  // namespace theLHCInfoPerFillImpl
class LHCInfoPerFillPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfoPerFill> {
public:
  LHCInfoPerFillPopConSourceHandler(edm::ParameterSet const& pset)
      : m_debug(pset.getUntrackedParameter<bool>("debug", false)),
        m_startTime(),
        m_endTime(),
        m_samplingInterval((unsigned int)pset.getUntrackedParameter<unsigned int>("samplingInterval", 300)),
        m_endFillMode(pset.getUntrackedParameter<bool>("endFill", true)),
        m_name(pset.getUntrackedParameter<std::string>("name", "LHCInfoPerFillPopConSourceHandler")),
        m_connectionString(pset.getUntrackedParameter<std::string>("connectionString", "")),
        m_ecalConnectionString(pset.getUntrackedParameter<std::string>("ecalConnectionString", "")),
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
  ~LHCInfoPerFillPopConSourceHandler() override = default;
  void getNewObjects() override {
    //reference to the last payload in the tag
    Ref previousFill;

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
    cond::persistency::Session session2 = connection.createSession(m_ecalConnectionString, false);
    // fetch last payload when available
    if (!tagInfo().lastInterval.payloadId.empty()) {
      cond::persistency::Session session3 = dbSession();
      session3.transaction().start(true);
      m_prevPayload = session3.fetchPayload<LHCInfoPerFill>(tagInfo().lastInterval.payloadId);
      session3.transaction().commit();
    }

    // bool iovAdded = false;
    while (true) {
      if (targetSince >= executionTimeIov) {
        edm::LogInfo(m_name) << "Sampling ended at the time "
                             << boost::posix_time::to_simple_string(cond::time::to_boost(executionTimeIov));
        break;
      }
      bool updateEcal = false;
      boost::posix_time::ptime targetTime = cond::time::to_boost(targetSince);
      boost::posix_time::ptime startSampleTime;
      boost::posix_time::ptime endSampleTime;

      cond::OMSService oms;
      oms.connect(m_omsBaseUrl);
      auto query = oms.query("fills");

      edm::LogInfo(m_name) << "Searching new fill after " << boost::posix_time::to_simple_string(targetTime);
      query->filterNotNull("start_stable_beam").filterNotNull("fill_number");
      if (targetTime > cond::time::to_boost(m_prevPayload->createTime())) {
        query->filterGE("start_time", targetTime);
      } else {
        query->filterGT("start_time", targetTime);
      }

      query->filterLT("start_time", m_endTime);
      if (m_endFillMode)
        query->filterNotNull("end_time");
      bool foundFill = query->execute();
      if (foundFill)
        foundFill = theLHCInfoPerFillImpl::makeFillPayload(m_fillPayload, query->result());
      if (!foundFill) {
        edm::LogInfo(m_name) << "No fill found - END of job.";
        // if (iovAdded)
        //   addEmptyPayload(targetSince);
        break;
      }

      startSampleTime = cond::time::to_boost(m_fillPayload->createTime());
      cond::Time_t startFillTime = m_fillPayload->createTime();
      cond::Time_t endFillTime = m_fillPayload->endTime();
      unsigned short lhcFill = m_fillPayload->fillNumber();
      bool ongoingFill = endFillTime == 0ULL;
      if (ongoingFill) {
        edm::LogInfo(m_name) << "Found ongoing fill " << lhcFill << " created at "
                             << cond::time::to_boost(startFillTime);
        endSampleTime = executionTime;
        targetSince = executionTimeIov;
      } else {
        edm::LogInfo(m_name) << "Found fill " << lhcFill << " created at " << cond::time::to_boost(startFillTime)
                             << " ending at " << cond::time::to_boost(endFillTime);
        endSampleTime = cond::time::to_boost(endFillTime);
        targetSince = endFillTime;
      }
      if (m_endFillMode || ongoingFill) {
        getDipData(oms, startSampleTime, endSampleTime);
        getLumiData(oms, lhcFill, startSampleTime, endSampleTime);
        if (!m_tmpBuffer.empty()) {
          boost::posix_time::ptime flumiStart = cond::time::to_boost(m_tmpBuffer.front().first);
          boost::posix_time::ptime flumiStop = cond::time::to_boost(m_tmpBuffer.back().first);
          edm::LogInfo(m_name) << "First lumi starts at " << flumiStart << " last lumi starts at " << flumiStop;
          session.transaction().start(true);
          getCTPPSData(session, startSampleTime, endSampleTime);
          session.transaction().commit();
          session2.transaction().start(true);
          getEcalData(session2, startSampleTime, endSampleTime, updateEcal);
          session2.transaction().commit();
        }
      }

      size_t niovs = theLHCInfoPerFillImpl::transferPayloads(m_tmpBuffer, m_iovs, m_prevPayload);
      edm::LogInfo(m_name) << "Added " << niovs << " iovs within the Fill time";
      m_tmpBuffer.clear();
      // iovAdded = true;
      if (m_prevPayload->fillNumber() and !ongoingFill)
        addEmptyPayload(endFillTime);
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
      auto newPayload = std::make_shared<LHCInfoPerFill>();
      m_iovs.insert(std::make_pair(iov, newPayload));
      m_prevPayload = newPayload;
      edm::LogInfo(m_name) << "Added empty payload with IOV " << iov << " ( "
                           << boost::posix_time::to_iso_extended_string(cond::time::to_boost(iov)) << " )";
    }
  }

  void addPayloadToBuffer(cond::OMSServiceResultRef& row) {
    auto lumiTime = row.get<boost::posix_time::ptime>("start_time");
    auto delivLumi = row.get<float>("delivered_lumi");
    auto recLumi = row.get<float>("recorded_lumi");
    LHCInfoPerFill* thisLumiSectionInfo = m_fillPayload->cloneFill();
    m_tmpBuffer.emplace_back(std::make_pair(cond::time::from_boost(lumiTime), thisLumiSectionInfo));
    LHCInfoPerFill& payload = *thisLumiSectionInfo;
    payload.setDelivLumi(delivLumi);
    payload.setRecLumi(recLumi);
  }

  size_t getLumiData(const cond::OMSService& oms,
                     unsigned short fillId,
                     const boost::posix_time::ptime& beginFillTime,
                     const boost::posix_time::ptime& endFillTime) {
    auto query = oms.query("lumisections");
    query->addOutputVars({"start_time", "delivered_lumi", "recorded_lumi", "beams_stable"});
    query->filterEQ("fill_number", fillId);
    query->filterGT("start_time", beginFillTime).filterLT("start_time", endFillTime);
    query->filterEQ("beams_stable", "true");
    query->limit(kLumisectionsQueryLimit);
    if (query->execute()) {
      auto queryResult = query->result();
      edm::LogInfo(m_name) << "Found " << queryResult.size() << " lumisections with STABLE BEAM during the fill "
                           << fillId;

      if (!queryResult.empty()) {
        if (m_endFillMode) {
          auto firstRow = queryResult.front();
          addPayloadToBuffer(firstRow);
        }

        auto lastRow = queryResult.back();
        addPayloadToBuffer(lastRow);
      }
    }
    return 0;
  }

  void getDipData(const cond::OMSService& oms,
                  const boost::posix_time::ptime& beginFillTime,
                  const boost::posix_time::ptime& endFillTime) {
    // unsure how to handle this.
    // the old implementation is not helping: apparently it is checking only the bunchconfiguration for the first diptime set of values...
    auto query1 = oms.query("diplogger/dip/acc/LHC/RunControl/CirculatingBunchConfig/Beam1");
    query1->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
    //This query is limited to 100 rows, but currently only one is used
    //If all this data is needed and saved properly the limit has to be set: query1->limit(...)
    if (query1->execute()) {
      auto res = query1->result();
      if (!res.empty()) {
        std::bitset<LHCInfoPerFill::bunchSlots + 1> bunchConfiguration1(0ULL);
        auto row = *res.begin();
        auto vbunchConf1 = row.getArray<unsigned short>("value");
        for (auto vb : vbunchConf1) {
          if (vb != 0) {
            unsigned short slot = (vb - 1) / 10 + 1;
            bunchConfiguration1[slot] = true;
          }
        }
        m_fillPayload->setBunchBitsetForBeam1(bunchConfiguration1);
      }
    }
    auto query2 = oms.query("diplogger/dip/acc/LHC/RunControl/CirculatingBunchConfig/Beam2");
    query2->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
    //This query is limited to 100 rows, but currently only one is used
    if (query2->execute()) {
      auto res = query2->result();
      if (!res.empty()) {
        std::bitset<LHCInfoPerFill::bunchSlots + 1> bunchConfiguration2(0ULL);
        auto row = *res.begin();
        auto vbunchConf2 = row.getArray<unsigned short>("value");
        for (auto vb : vbunchConf2) {
          if (vb != 0) {
            unsigned short slot = (vb - 1) / 10 + 1;
            bunchConfiguration2[slot] = true;
          }
        }
        m_fillPayload->setBunchBitsetForBeam2(bunchConfiguration2);
      }
    }

    auto query3 = oms.query("diplogger/dip/CMS/LHC/LumiPerBunch");
    query3->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
    //This query is limited to 100 rows, but currently only one is used
    if (query3->execute()) {
      auto res = query3->result();
      if (!res.empty()) {
        std::vector<float> lumiPerBX;
        auto row = *res.begin();
        auto lumiBunchInst = row.getArray<float>("lumi_bunch_inst");
        for (auto lb : lumiBunchInst) {
          if (lb != 0.) {
            lumiPerBX.push_back(lb);
          }
        }
        m_fillPayload->setLumiPerBX(lumiPerBX);
      }
    }
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
    CTPPSDataQuery->addToOutputList(std::string("LHC_STATE"));
    CTPPSDataQuery->addToOutputList(std::string("LHC_COMMENT"));
    if (m_debug) {
      CTPPSDataQuery->addToOutputList(std::string("RUN_NUMBER"));
      CTPPSDataQuery->addToOutputList(std::string("LUMI_SECTION"));
    }
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
    CTPPSDataOutput.extend<std::string>(std::string("LHC_STATE"));
    CTPPSDataOutput.extend<std::string>(std::string("LHC_COMMENT"));
    if (m_debug) {
      CTPPSDataOutput.extend<int>(std::string("RUN_NUMBER"));
      CTPPSDataOutput.extend<int>(std::string("LUMI_SECTION"));
    }
    CTPPSDataQuery->defineOutput(CTPPSDataOutput);
    //execute the query
    coral::ICursor& CTPPSDataCursor = CTPPSDataQuery->execute();
    cond::Time_t dipTime = 0;
    std::string lhcState = "", lhcComment = "", ctppsStatus = "";

    //debug informations
    unsigned int lumiSection = 0;
    cond::Time_t runNumber = 0;
    cond::Time_t savedDipTime = 0;
    unsigned int savedLumiSection = 0;
    cond::Time_t savedRunNumber = 0;

    bool ret = false;
    LumiSectionFilter<LHCInfoPerFill> filter(m_tmpBuffer);
    while (CTPPSDataCursor.next()) {
      if (m_debug) {
        std::ostringstream CTPPS;
        CTPPSDataCursor.currentRow().toOutputStream(CTPPS);
      }
      coral::Attribute const& dipTimeAttribute = CTPPSDataCursor.currentRow()[std::string("DIP_UPDATE_TIME")];
      if (!dipTimeAttribute.isNull()) {
        dipTime = cond::time::from_boost(dipTimeAttribute.data<coral::TimeStamp>().time());
        if (filter.process(dipTime)) {
          ret = true;
          coral::Attribute const& lhcStateAttribute = CTPPSDataCursor.currentRow()[std::string("LHC_STATE")];
          if (!lhcStateAttribute.isNull()) {
            lhcState = lhcStateAttribute.data<std::string>();
          }
          coral::Attribute const& lhcCommentAttribute = CTPPSDataCursor.currentRow()[std::string("LHC_COMMENT")];
          if (!lhcCommentAttribute.isNull()) {
            lhcComment = lhcCommentAttribute.data<std::string>();
          }

          if (m_debug) {
            coral::Attribute const& runNumberAttribute = CTPPSDataCursor.currentRow()[std::string("RUN_NUMBER")];
            if (!runNumberAttribute.isNull()) {
              runNumber = runNumberAttribute.data<int>();
            }
            coral::Attribute const& lumiSectionAttribute = CTPPSDataCursor.currentRow()[std::string("LUMI_SECTION")];
            if (!lumiSectionAttribute.isNull()) {
              lumiSection = lumiSectionAttribute.data<int>();
            }
          }

          for (auto it = filter.current(); it != m_tmpBuffer.end(); it++) {
            // set the current values to all of the payloads of the lumi section samples after the current since
            LHCInfoPerFill& payload = *(it->second);
            payload.setLhcState(lhcState);
            payload.setLhcComment(lhcComment);
            payload.setCtppsStatus(ctppsStatus);

            if (m_debug) {
              savedDipTime = dipTime;
              savedLumiSection = lumiSection;
              savedRunNumber = runNumber;
            }
          }
        }
      }
    }
    if (m_debug) {
      edm::LogInfo(m_name) << "Last assigned: "
                           << "DipTime: " << savedDipTime << " "
                           << "LumiSection: " << savedLumiSection << " "
                           << "RunNumber: " << savedRunNumber;
    }
    return ret;
  }

  bool getEcalData(cond::persistency::Session& session,
                   const boost::posix_time::ptime& lowerTime,
                   const boost::posix_time::ptime& upperTime,
                   bool update) {
    //run the sixth query against the CMS_DCS_ENV_PVSS_COND schema
    //Initializing the CMS_DCS_ENV_PVSS_COND schema.
    coral::ISchema& ECAL = session.nominalSchema();
    //start the transaction against the fill logging schema
    //execute query for ECAL Data
    std::unique_ptr<coral::IQuery> ECALDataQuery(ECAL.newQuery());
    //FROM clause
    ECALDataQuery->addToTableList(std::string("BEAM_PHASE"));
    //SELECT clause
    ECALDataQuery->addToOutputList(std::string("CHANGE_DATE"));
    ECALDataQuery->addToOutputList(std::string("DIP_value"));
    ECALDataQuery->addToOutputList(std::string("element_nr"));
    ECALDataQuery->addToOutputList(std::string("VALUE_NUMBER"));
    //WHERE CLAUSE
    coral::AttributeList ECALDataBindVariables;
    ECALDataBindVariables.extend<coral::TimeStamp>(std::string("lowerTime"));
    ECALDataBindVariables.extend<coral::TimeStamp>(std::string("upperTime"));
    ECALDataBindVariables[std::string("lowerTime")].data<coral::TimeStamp>() = coral::TimeStamp(lowerTime);
    ECALDataBindVariables[std::string("upperTime")].data<coral::TimeStamp>() = coral::TimeStamp(upperTime);
    std::string conditionStr = std::string(
        "(DIP_value LIKE '%beamPhaseMean%' OR DIP_value LIKE '%cavPhaseMean%') AND CHANGE_DATE >= :lowerTime AND "
        "CHANGE_DATE < :upperTime");

    ECALDataQuery->setCondition(conditionStr, ECALDataBindVariables);
    //ORDER BY clause
    ECALDataQuery->addToOrderList(std::string("CHANGE_DATE"));
    ECALDataQuery->addToOrderList(std::string("DIP_value"));
    ECALDataQuery->addToOrderList(std::string("element_nr"));
    //define query output
    coral::AttributeList ECALDataOutput;
    ECALDataOutput.extend<coral::TimeStamp>(std::string("CHANGE_DATE"));
    ECALDataOutput.extend<std::string>(std::string("DIP_value"));
    ECALDataOutput.extend<unsigned int>(std::string("element_nr"));
    ECALDataOutput.extend<float>(std::string("VALUE_NUMBER"));
    //ECALDataQuery->limitReturnedRows( 14256 ); //3564 entries per vector.
    ECALDataQuery->defineOutput(ECALDataOutput);
    //execute the query
    coral::ICursor& ECALDataCursor = ECALDataQuery->execute();
    cond::Time_t changeTime = 0;
    cond::Time_t firstTime = 0;
    std::string dipVal = "";
    unsigned int elementNr = 0;
    float value = 0.;
    std::set<cond::Time_t> initializedVectors;
    LumiSectionFilter<LHCInfoPerFill> filter(m_tmpBuffer);
    bool ret = false;
    if (m_prevPayload.get()) {
      for (auto& lumiSlot : m_tmpBuffer) {
        lumiSlot.second->setBeam1VC(m_prevPayload->beam1VC());
        lumiSlot.second->setBeam2VC(m_prevPayload->beam2VC());
        lumiSlot.second->setBeam1RF(m_prevPayload->beam1RF());
        lumiSlot.second->setBeam2RF(m_prevPayload->beam2RF());
      }
    }
    std::map<cond::Time_t, cond::Time_t> iovMap;
    if (m_tmpBuffer.empty()) {
      return ret;
    }
    cond::Time_t lowerLumi = m_tmpBuffer.front().first;
    while (ECALDataCursor.next()) {
      if (m_debug) {
        std::ostringstream ECAL;
        ECALDataCursor.currentRow().toOutputStream(ECAL);
      }
      coral::Attribute const& changeDateAttribute = ECALDataCursor.currentRow()[std::string("CHANGE_DATE")];
      if (!changeDateAttribute.isNull()) {
        ret = true;
        boost::posix_time::ptime chTime = changeDateAttribute.data<coral::TimeStamp>().time();
        // move the first IOV found to the start of the fill interval selected
        if (changeTime == 0) {
          firstTime = cond::time::from_boost(chTime);
        }
        changeTime = cond::time::from_boost(chTime);
        cond::Time_t iovTime = changeTime;
        if (!update and changeTime == firstTime)
          iovTime = lowerLumi;
        coral::Attribute const& dipValAttribute = ECALDataCursor.currentRow()[std::string("DIP_value")];
        coral::Attribute const& valueNumberAttribute = ECALDataCursor.currentRow()[std::string("VALUE_NUMBER")];
        coral::Attribute const& elementNrAttribute = ECALDataCursor.currentRow()[std::string("element_nr")];
        if (!dipValAttribute.isNull() and !valueNumberAttribute.isNull()) {
          dipVal = dipValAttribute.data<std::string>();
          elementNr = elementNrAttribute.data<unsigned int>();
          value = valueNumberAttribute.data<float>();
          if (std::isnan(value))
            value = 0.;
          if (filter.process(iovTime)) {
            iovMap.insert(std::make_pair(changeTime, filter.current()->first));
            for (auto it = filter.current(); it != m_tmpBuffer.end(); it++) {
              LHCInfoPerFill& payload = *(it->second);
              theLHCInfoPerFillImpl::setElementData(it->first, dipVal, elementNr, value, payload, initializedVectors);
            }
          }
          //}
        }
      }
    }
    if (m_debug) {
      for (auto& im : iovMap) {
        edm::LogInfo(m_name) << "Found iov=" << im.first << " (" << cond::time::to_boost(im.first) << " ) moved to "
                             << im.second << " ( " << cond::time::to_boost(im.second) << " )";
      }
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
  std::unique_ptr<LHCInfoPerFill> m_fillPayload;
  std::shared_ptr<LHCInfoPerFill> m_prevPayload;
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>> m_tmpBuffer;
  bool m_lastPayloadEmpty = false;
};
