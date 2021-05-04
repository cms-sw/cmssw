#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondTools/BeamSpot/interface/BeamSpotOnlinePopConSourceHandler.h"

#include <chrono>
#include <memory>

BeamSpotOnlinePopConSourceHandler::BeamSpotOnlinePopConSourceHandler(edm::ParameterSet const& pset)
    : m_debug(pset.getUntrackedParameter<bool>("debug", false)),
      m_name(pset.getUntrackedParameter<std::string>("name", "BeamSpotOnlineSourceHandler")),
      m_maxAge(pset.getUntrackedParameter<unsigned int>("maxAge", 86400)),
      m_runNumber(pset.getUntrackedParameter<unsigned int>("runNumber", 1)),
      m_sourcePayloadTag(pset.getUntrackedParameter<std::string>("sourcePayloadTag", "")) {}

BeamSpotOnlinePopConSourceHandler::~BeamSpotOnlinePopConSourceHandler() {}

bool checkPayloadAge(const BeamSpotOnlineObjects& payload, unsigned int maxAge) {
  long creationTimeInSeconds = payload.GetCreationTime() >> 32;
  auto timeNow = std::chrono::system_clock::now();
  auto nowSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(timeNow.time_since_epoch()).count();
  long age = nowSinceEpoch - creationTimeInSeconds;
  return age < maxAge;
}

std::unique_ptr<BeamSpotOnlineObjects> makeDummyPayload() {
  // implement here
  std::unique_ptr<BeamSpotOnlineObjects> ret;
  ret = std::make_unique<BeamSpotOnlineObjects>();
  auto timeNow = std::chrono::system_clock::now();
  auto nowSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(timeNow.time_since_epoch()).count();
  ret->SetCreationTime(nowSinceEpoch << 32);
  return ret;
}

void BeamSpotOnlinePopConSourceHandler::getNewObjects() {
  bool addNewPayload = false;
  if (!tagInfo().size) {
    edm::LogInfo(m_name) << "New tag " << tagInfo().name << "; from " << m_name << "::getNewObjects";
    addNewPayload = true;
  } else {
    edm::LogInfo(m_name) << "got info for tag " << tagInfo().name << ", last object valid since "
                         << tagInfo().lastInterval.since << "; from " << m_name << "::getNewObjects";
    if (!checkPayloadAge(*lastPayload(), m_maxAge)) {
      addNewPayload = true;
    }
  }

  if (addNewPayload) {
    if (!m_sourcePayloadTag.empty()) {
      edm::LogInfo(m_name) << "Reading target payload from tag " << m_sourcePayloadTag;
      auto session = dbSession();
      session.transaction().start(true);
      auto lastIov = session.readIov(m_sourcePayloadTag).getLast();
      m_payload = session.fetchPayload<BeamSpotOnlineObjects>(lastIov.payloadId);
      session.transaction().commit();
    } else {
      m_payload = makeDummyPayload();
    }

    cond::Time_t targetTime = cond::time::lumiTime(m_runNumber, 1);
    m_to_transfer.push_back(std::make_pair(m_payload.get(), targetTime));

    edm::LogInfo(m_name) << "Payload added with IOV since " << targetTime;
  } else {
    edm::LogInfo(m_name) << "Nothing to do, last payload satisfies maximum age requirement.";
  }
}

std::string BeamSpotOnlinePopConSourceHandler::id() const { return m_name; }
