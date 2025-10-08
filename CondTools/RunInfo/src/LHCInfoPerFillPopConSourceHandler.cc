#include "CondTools/RunInfo/interface/LHCInfoPerFillPopConSourceHandler.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondTools/RunInfo/interface/LumiSectionFilter.h"
#include "CondTools/RunInfo/interface/LHCInfoHelper.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"

using std::make_pair;
using std::pair;

namespace cond {
  namespace theLHCInfoPerFillPopConImpl {

    static const pair<const char*, LHCInfoPerFill::FillType> s_fillTypeMap[] = {
        make_pair("PROTONS", LHCInfoPerFill::PROTONS),
        make_pair("IONS", LHCInfoPerFill::IONS),
        make_pair("COSMICS", LHCInfoPerFill::COSMICS),
        make_pair("GAP", LHCInfoPerFill::GAP)};

    static const pair<const char*, LHCInfoPerFill::ParticleType> s_particleTypeMap[] = {
        make_pair("PROTON", LHCInfoPerFill::PROTON),
        make_pair("PB82", LHCInfoPerFill::PB82),
        make_pair("AR18", LHCInfoPerFill::AR18),
        make_pair("D", LHCInfoPerFill::D),
        make_pair("XE54", LHCInfoPerFill::XE54)};

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

      if (energy <= 0) {
        // only log an error, do not fail the payload creation, the logic of skipping payloads with invalid energy is handled elsewhere
        edm::LogError("LHCInfoPerFillPopConSourceHandler")
            << "Received non-positive energy from OMS for fill " << currentFill << ": " << energy
            << " GeV, string value: '" << row.get<std::string>("energy") << "'.";
      }
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

  size_t transferPayloads(const std::vector<pair<cond::Time_t, std::shared_ptr<LHCInfoPerFill>>>& buffer,
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
        iovsToTransfer.insert(make_pair(since, payload));
        prevPayload = iov.second;
      }
    }
    edm::LogInfo("transferPayloads") << "TRANSFERED IOVS: " << condIovs.str();
    edm::LogInfo("transferPayloads") << "FORMATTED TRANSFERED IOVS: " << formattedIovs.str();
    return niovs;
  }

}  // namespace theLHCInfoPerFillImpl

LHCInfoPerFillPopConSourceHandler::LHCInfoPerFillPopConSourceHandler(edm::ParameterSet const& pset)
    : m_debug(pset.getUntrackedParameter<bool>("debug", false)),
      m_startTime(),
      m_endTime(),
      m_endFillMode(pset.getUntrackedParameter<bool>("endFill", true)),
      m_name(pset.getUntrackedParameter<std::string>("name", "LHCInfoPerFillPopConSourceHandler")),
      m_connectionString(pset.getUntrackedParameter<std::string>("connectionString", "")),
      m_ecalConnectionString(pset.getUntrackedParameter<std::string>("ecalConnectionString", "")),
      m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath", "")),
      m_omsBaseUrl(pset.getUntrackedParameter<std::string>("omsBaseUrl", "")),
      m_minEnergy(pset.getUntrackedParameter<double>("minEnergy", 450.)),
      m_maxEnergy(pset.getUntrackedParameter<double>("maxEnergy", 8000.)),
      m_throwOnInvalid(pset.getUntrackedParameter<bool>("throwOnInvalid", false)),
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

void LHCInfoPerFillPopConSourceHandler::getNewObjects() {
  populateIovs();
  if (!m_endFillMode) {  // duringFill mode
    handleInvalidPayloads();
  }
}

void LHCInfoPerFillPopConSourceHandler::handleInvalidPayloads() {
  // note: at the moment used only in duringFill mode so the m_iovs is quaranteed to have size() <= 1
  // but iterating through the whole map is implemented just in case the way it's used changes
  auto it = m_iovs.begin();
  while (it != m_iovs.end()) {
    std::stringstream payloadData;
    payloadData << "Fill = " << it->second->fillNumber() << ", Energy = " << it->second->energy();
    if (!isPayloadValid(*(it->second))) {
      // define the message and then either throw or print log and filter out
      std::string msg = "Skipping upload of payload with invalid values: " + payloadData.str();
      if (m_throwOnInvalid) {
        throw cms::Exception("LHCInfoPerFillPopConSourceHandler") << msg;
      } else {
        edm::LogWarning(m_name) << msg;
      }
      // filter out (erase) invalid payloads
      m_iovs.erase(it++);  // note: post-increment necessary to avoid using invalidated iterators
    } else {
      edm::LogInfo(m_name) << "Payload to be uploaded: " << payloadData.str();
      ++it;
    }
  }
}

bool LHCInfoPerFillPopConSourceHandler::isPayloadValid(const LHCInfoPerFill& payload) const {
  return (m_minEnergy <= payload.energy() && payload.energy() <= m_maxEnergy);
}

std::tuple<cond::persistency::Session, cond::persistency::Session>
LHCInfoPerFillPopConSourceHandler::createSubsystemDbSessions() const {
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
  cond::persistency::Session cttpsSession = connection.createSession(m_connectionString, false);
  cond::persistency::Session ecalSession = connection.createSession(m_ecalConnectionString, false);
  return std::make_tuple(std::move(cttpsSession), std::move(ecalSession));
}

cond::Time_t LHCInfoPerFillPopConSourceHandler::getNextFillSearchTimestamp(cond::Time_t lastSince) const {
  cond::Time_t startTimestamp = m_startTime.is_not_a_date_time() ? 0 : cond::time::from_boost(m_startTime);
  cond::Time_t nextFillSearchTimestamp =
      std::max(startTimestamp, m_endFillMode ? lastSince : (m_prevPayload ? m_prevPayload->createTime() : 0));
  return nextFillSearchTimestamp;
}

cond::Time_t LHCInfoPerFillPopConSourceHandler::handleIfNewTagAndGetLastSince() {
  //print tag info
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
    if (m_endFillMode) {
      // for a new or empty tag in endFill mode, an empty payload should be added on top with since=1
      addEmptyPayload(1);
      lastSince = 1;
    } else {
      // in duringFill mode, we don't upload empty payloads to the empty tag
      lastSince = 0;  // in duringFill mode, this value is not used when the tag is empty
    }
  } else {
    edm::LogInfo(m_name) << "The last Iov in tag " << tagInfo().name << " valid since " << lastSince << "from "
                         << m_name << "::getNewObjects";
  }

  return lastSince;
}

void LHCInfoPerFillPopConSourceHandler::fetchLastPayload() {
  if (!tagInfo().lastInterval.payloadId.empty()) {
    cond::persistency::Session session3 = dbSession();
    session3.transaction().start(true);
    m_prevPayload = session3.fetchPayload<LHCInfoPerFill>(tagInfo().lastInterval.payloadId);
    session3.transaction().commit();
  }
}

boost::posix_time::ptime LHCInfoPerFillPopConSourceHandler::getExecutionTime() const {
  return boost::posix_time::second_clock::local_time();
}

void LHCInfoPerFillPopConSourceHandler::populateIovs() {
  cond::Time_t lastSince = handleIfNewTagAndGetLastSince();
  fetchLastPayload();
  cond::Time_t nextFillSearchTimestamp = getNextFillSearchTimestamp(lastSince);
  edm::LogInfo(m_name) << "Starting sampling at "
                       << boost::posix_time::to_simple_string(cond::time::to_boost(nextFillSearchTimestamp));

  auto [cttpsSession, ecalSession] = createSubsystemDbSessions();
  boost::posix_time::ptime executionTime = getExecutionTime();
  cond::Time_t executionTimeIov = cond::time::from_boost(executionTime);

  while (true) {
    if (nextFillSearchTimestamp >= executionTimeIov) {
      edm::LogInfo(m_name) << "Sampling ended at the time "
                           << boost::posix_time::to_simple_string(cond::time::to_boost(executionTimeIov));
      break;
    }
    boost::posix_time::ptime nextFillSearchTime = cond::time::to_boost(nextFillSearchTimestamp);
    boost::posix_time::ptime startSampleTime;
    boost::posix_time::ptime endSampleTime;

    cond::OMSService oms;
    oms.connect(m_omsBaseUrl);

    bool inclusiveSearchTime =
        nextFillSearchTime > cond::time::to_boost(m_prevPayload ? m_prevPayload->createTime() : 0);
    m_fillPayload = findFillToProcess(oms, nextFillSearchTime, inclusiveSearchTime);

    if (!m_fillPayload) {
      edm::LogInfo(m_name) << "No fill found - END of job.";
      break;
    }

    startSampleTime = cond::time::to_boost(m_fillPayload->createTime());
    cond::Time_t startFillTime = m_fillPayload->createTime();
    cond::Time_t endFillTime = m_fillPayload->endTime();
    unsigned short lhcFill = m_fillPayload->fillNumber();
    bool ongoingFill = endFillTime == 0ULL;
    if (ongoingFill) {
      edm::LogInfo(m_name) << "Found ongoing fill " << lhcFill << " created at " << cond::time::to_boost(startFillTime);
      endSampleTime = executionTime;
      nextFillSearchTimestamp = executionTimeIov;
    } else {
      edm::LogInfo(m_name) << "Found fill " << lhcFill << " created at " << cond::time::to_boost(startFillTime)
                           << " ending at " << cond::time::to_boost(endFillTime);
      endSampleTime = cond::time::to_boost(endFillTime);
      nextFillSearchTimestamp = endFillTime;
    }
    if (m_endFillMode || ongoingFill) {
      getDipData(oms, startSampleTime, endSampleTime);
      getLumiData(oms, lhcFill, startSampleTime, endSampleTime);
      if (!m_tmpBuffer.empty()) {
        boost::posix_time::ptime flumiStart = cond::time::to_boost(m_tmpBuffer.front().first);
        boost::posix_time::ptime flumiStop = cond::time::to_boost(m_tmpBuffer.back().first);
        edm::LogInfo(m_name) << "First lumi starts at " << flumiStart << " last lumi starts at " << flumiStop;
        getCTPPSData(cttpsSession, startSampleTime, endSampleTime);
        getEcalData(ecalSession, startSampleTime, endSampleTime);
      }
    }

    if (!m_endFillMode) {
      if (m_tmpBuffer.size() > 1) {
        throw cms::Exception("LHCInfoPerFillPopConSourceHandler")
            << "More than 1 payload buffered for writing in duringFill mode.\
          In this mode only up to 1 payload can be written";
      } else if (m_tmpBuffer.size() == 1) {
        if (m_prevPayload && theLHCInfoPerFillImpl::comparePayloads(*(m_tmpBuffer.begin()->second), *m_prevPayload)) {
          m_tmpBuffer.clear();
          edm::LogInfo(m_name)
              << "The buffered payload has the same data as the previous payload in the tag. It will not be written.";
        }
      }
      // In duringFill mode, convert the timestamp-type IOVs to lumiid-type IOVs
      // before transferring the payloads from the buffer to the final collection
      convertBufferedIovsToLumiid(m_timestampToLumiid);
    }

    size_t niovs = theLHCInfoPerFillImpl::transferPayloads(m_tmpBuffer, m_iovs, m_prevPayload);
    edm::LogInfo(m_name) << "Added " << niovs << " iovs within the Fill time";
    m_tmpBuffer.clear();
    m_timestampToLumiid.clear();

    if (!m_endFillMode) {
      return;
    }

    // endFill mode only:
    if (m_prevPayload->fillNumber() and !ongoingFill) {
      if (m_endFillMode) {
        addEmptyPayload(endFillTime);
      }
    }
  }
}

std::string LHCInfoPerFillPopConSourceHandler::id() const { return m_name; }

std::unique_ptr<LHCInfoPerFill> LHCInfoPerFillPopConSourceHandler::findFillToProcess(
    cond::OMSService& oms, const boost::posix_time::ptime& nextFillSearchTime, bool inclusiveSearchTime) {
  oms.connect(m_omsBaseUrl);
  auto query = oms.query("fills");

  edm::LogInfo(m_name) << "Searching new fill after " << boost::posix_time::to_simple_string(nextFillSearchTime);
  query->filterNotNull("start_stable_beam").filterNotNull("fill_number");
  if (inclusiveSearchTime) {
    query->filterGE("start_time", nextFillSearchTime);
  } else {
    query->filterGT("start_time", nextFillSearchTime);
  }

  query->filterLT("start_time", m_endTime);
  if (m_endFillMode)
    query->filterNotNull("end_time");
  else
    query->filterEQ("end_time", cond::OMSServiceQuery::SNULL);

  bool foundFill = query->execute();
  std::unique_ptr<LHCInfoPerFill> fillToBeProcessedPayload;
  if (foundFill)
    foundFill = theLHCInfoPerFillImpl::makeFillPayload(fillToBeProcessedPayload, query->result());
  return fillToBeProcessedPayload;
}

void LHCInfoPerFillPopConSourceHandler::addEmptyPayload(cond::Time_t iov) {
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
    m_iovs.insert(make_pair(iov, newPayload));
    m_prevPayload = newPayload;
    edm::LogInfo(m_name) << "Added empty payload with IOV " << iov << " ( "
                         << boost::posix_time::to_iso_extended_string(cond::time::to_boost(iov)) << " )";
  }
}

// Add payload to buffer and store corresponding lumiid IOV in m_timestampToLumiid map
void LHCInfoPerFillPopConSourceHandler::addPayloadToBuffer(cond::OMSServiceResultRef& row) {
  auto startTime = row.get<boost::posix_time::ptime>("start_time");
  auto delivLumi = row.get<float>("delivered_lumi");
  auto recLumi = row.get<float>("recorded_lumi");
  auto runNumber = std::stoul(row.get<std::string>("run_number"));
  auto lsNumber = std::stoul(row.get<std::string>("lumisection_number"));
  auto lumiid = cond::time::lumiTime(runNumber, lsNumber);

  LHCInfoPerFill* thisLumiSectionInfo = m_fillPayload->cloneFill();
  m_tmpBuffer.emplace_back(make_pair(cond::time::from_boost(startTime), thisLumiSectionInfo));
  if (!m_endFillMode) {
    m_timestampToLumiid.insert(make_pair(cond::time::from_boost(startTime), lumiid));
  }
  LHCInfoPerFill& payload = *thisLumiSectionInfo;
  payload.setDelivLumi(delivLumi);
  payload.setRecLumi(recLumi);
}

void LHCInfoPerFillPopConSourceHandler::convertBufferedIovsToLumiid(
    std::map<cond::Time_t, cond::Time_t> timestampToLumiid) {
  for (auto& item : m_tmpBuffer) {
    // Check if the lumiid IOV corresponding to the timestamp is present in the map
    if (timestampToLumiid.find(item.first) == timestampToLumiid.end()) {
      throw cms::Exception("LHCInfoPerFillPopConSourceHandler")
          << "Can't find corresponding lumiid IOV for timestamp " << item.first << "\n";
    }
    // Update the buffer with the lumiid-type IOV
    item.first = timestampToLumiid.at(item.first);
  }
}

std::tuple<cond::OMSServiceResult, bool, std::unique_ptr<cond::OMSServiceQuery>>
LHCInfoPerFillPopConSourceHandler::executeLumiQuery(const cond::OMSService& oms,
                                                    unsigned short fillId,
                                                    const boost::posix_time::ptime& beginFillTime,
                                                    const boost::posix_time::ptime& endFillTime) const {
  auto query = oms.query("lumisections");
  query->addOutputVars(
      {"start_time", "delivered_lumi", "recorded_lumi", "beams_stable", "run_number", "lumisection_number"});
  query->filterEQ("fill_number", fillId);
  query->filterGT("start_time", beginFillTime).filterLT("start_time", endFillTime);
  query->filterEQ("beams_stable", "true");
  query->limit(cond::lhcInfoHelper::kLumisectionsQueryLimit);

  bool success = query->execute();
  return {(success ? query->result() : cond::OMSServiceResult()), success, std::move(query)};
}

void LHCInfoPerFillPopConSourceHandler::getLumiData(const cond::OMSService& oms,
                                                    unsigned short fillId,
                                                    const boost::posix_time::ptime& beginFillTime,
                                                    const boost::posix_time::ptime& endFillTime) {
  // keeping the query ownerObject in scope is necessary as the lifetime of the queryResult resources is tied to it
  auto [queryResult, success, ownerObject] = executeLumiQuery(oms, fillId, beginFillTime, endFillTime);
  if (!success) {
    edm::LogError(m_name) << "Failed to execute luminosity query.";
    return;
  }
  edm::LogInfo(m_name) << "Found " << queryResult.size() << " lumisections with STABLE BEAM during the fill " << fillId;

  if (queryResult.empty()) {
    edm::LogWarning(m_name) << "No lumisections with STABLE BEAM found during the fill " << fillId
                            << ". No payload will be added to buffer for writing.";
    return;
  }

  if (m_endFillMode) {
    auto firstRow = queryResult.front();
    addPayloadToBuffer(firstRow);
  }

  auto lastRow = queryResult.back();
  addPayloadToBuffer(lastRow);

  return;
}

void LHCInfoPerFillPopConSourceHandler::getDipData(const cond::OMSService& oms,
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

bool LHCInfoPerFillPopConSourceHandler::getCTPPSData(cond::persistency::Session& cttpsSession,
                                                     const boost::posix_time::ptime& beginFillTime,
                                                     const boost::posix_time::ptime& endFillTime) {
  cttpsSession.transaction().start(true);
  auto ret = getCTPPSDataImpl(cttpsSession, beginFillTime, endFillTime);
  cttpsSession.transaction().commit();
  return ret;
}

bool LHCInfoPerFillPopConSourceHandler::getCTPPSDataImpl(cond::persistency::Session& cttpsSession,
                                                         const boost::posix_time::ptime& beginFillTime,
                                                         const boost::posix_time::ptime& endFillTime) {
  //run the fifth query against the CTPPS schema
  //Initializing the CMS_CTP_CTPPS_COND schema.
  coral::ISchema& CTPPS = cttpsSession.coralSession().schema("CMS_PPS_SPECT_COND");
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

bool LHCInfoPerFillPopConSourceHandler::getEcalData(cond::persistency::Session& ecalSession,
                                                    const boost::posix_time::ptime& lowerTime,
                                                    const boost::posix_time::ptime& upperTime) {
  ecalSession.transaction().start(true);
  auto ret = getEcalDataImpl(ecalSession, lowerTime, upperTime);
  ecalSession.transaction().commit();
  return ret;
}

bool LHCInfoPerFillPopConSourceHandler::getEcalDataImpl(cond::persistency::Session& ecalSession,
                                                        const boost::posix_time::ptime& lowerTime,
                                                        const boost::posix_time::ptime& upperTime) {
  //run the sixth query against the CMS_DCS_ENV_PVSS_COND schema
  //Initializing the CMS_DCS_ENV_PVSS_COND schema.
  coral::ISchema& ECAL = ecalSession.nominalSchema();
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
      if (changeTime == firstTime)
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
          iovMap.insert(make_pair(changeTime, filter.current()->first));
          for (auto it = filter.current(); it != m_tmpBuffer.end(); it++) {
            LHCInfoPerFill& payload = *(it->second);
            theLHCInfoPerFillImpl::setElementData(it->first, dipVal, elementNr, value, payload, initializedVectors);
          }
        }
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
