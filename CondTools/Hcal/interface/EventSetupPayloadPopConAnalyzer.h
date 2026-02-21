#pragma once

#include <memory>
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"

template <typename T>
concept CanAcceptPayload = requires(T& h, std::unique_ptr<typename T::value_type> p) {
  typename T::value_type;
  { h.initPayload(std::move(p)) };
};

template <CanAcceptPayload SourceHandler, typename PayloadRcd>
class EventSetupPayloadPopConAnalyzer : public popcon::PopConAnalyzer<SourceHandler> {
public:
  using Payload = typename SourceHandler::value_type;

  EventSetupPayloadPopConAnalyzer(const edm::ParameterSet& pset)
      : popcon::PopConAnalyzer<SourceHandler>(pset),
        m_token(this->template esConsumes<Payload, PayloadRcd>()),
        // false as default as most/all relevant PopCon workflows now assume running analyze only once
        m_allowMultipleAnalyze(pset.getUntrackedParameter<bool>("allowMultipleAnalyze", false)) {}

private:
  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    if (m_analyzed && !m_allowMultipleAnalyze) {
      throw cms::Exception("EventSetupPayloadPopConAnalyzer")
          << "analyze called multiple times which is not inteded for this object: m_allowMultipleAnalyze="
          << m_allowMultipleAnalyze;
    }

    //Using ES to get the data:
    this->source().initPayload(std::make_unique<Payload>(esetup.getData(m_token)));
    m_analyzed = true;
  }

private:
  edm::ESGetToken<Payload, PayloadRcd> m_token;
  bool m_allowMultipleAnalyze;
  bool m_analyzed = false;
};
