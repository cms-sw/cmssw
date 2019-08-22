#ifndef ECALPFRECHITTHRESHOLDSMAKER_H
#define ECALPFRECHITTHRESHOLDSMAKER_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalPFRecHitThresholdsMaker : public edm::EDAnalyzer {
public:
  explicit EcalPFRecHitThresholdsMaker(const edm::ParameterSet& iConfig);
  ~EcalPFRecHitThresholdsMaker() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  std::string m_timetype;
  double m_nsigma;
};

#endif
