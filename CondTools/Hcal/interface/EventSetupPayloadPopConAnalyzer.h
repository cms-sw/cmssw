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
      : popcon::PopConAnalyzer<SourceHandler>(pset), m_token(this->template esConsumes<Payload, PayloadRcd>()) {}

private:
  void analyze(const edm::Event& ev, const edm::EventSetup& esetup) override {
    // TODO confirm that all use cases are meant to have this run only once:
    // then throw exception if run more than once
    // and rename the class to include "Single"

    //Using ES to get the data:
    this->source().initPayload(std::make_unique<Payload>(esetup.getData(m_token)));
  }

private:
  edm::ESGetToken<Payload, PayloadRcd> m_token;
};
