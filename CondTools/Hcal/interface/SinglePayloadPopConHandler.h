#pragma once

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <memory>
#include <string>
#include <typeinfo>

template <typename Payload>
class SinglePayloadPopConHandler : public popcon::PopConSourceHandler<Payload> {
public:
  SinglePayloadPopConHandler(edm::ParameterSet const& ps)
      : m_name(ps.getUntrackedParameter<std::string>(
            "name", std::string("SinglePayloadPopConHandler<") + typeid(Payload).name() + ">")),
        m_sinceTime(ps.getUntrackedParameter<unsigned>("IOVRun", 0)) {}

  void getNewObjects() override {
    edm::LogInfo(m_name) << "------- " << m_name << " - > getNewObjects\n"
                         << "got offlineInfo " << this->tagInfo().name << ", size " << this->tagInfo().size
                         << ", last object valid since " << this->tagInfo().lastInterval.since;

    if (!m_payload) {
      throw cms::Exception("Empty DB object") << m_name << " has received empty object - nothing to write to DB";
    }

    edm::LogInfo(m_name) << "Using IOV run " << m_sinceTime;

    // prepare for transfer:
    this->m_iovs.emplace(m_sinceTime, std::shared_ptr<Payload>(std::move(m_payload)));

    edm::LogInfo(m_name) << "------- " << m_name << " - > getNewObjects" << std::endl;
  }

  std::string id() const override { return m_name; }

  void initPayload(std::unique_ptr<Payload> payload) { m_payload = std::move(payload); }

private:
  std::string m_name;
  unsigned int m_sinceTime;
  std::unique_ptr<Payload> m_payload;
};
