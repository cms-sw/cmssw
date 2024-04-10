#ifndef CalibCalorimetry_HcalPlugins_HcalTimeSlewEP_H
#define CalibCalorimetry_HcalPlugins_HcalTimeSlewEP_H

// system include files
#include <memory>

//user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HcalTimeSlewEP : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HcalTimeSlewEP(const edm::ParameterSet&);
  ~HcalTimeSlewEP() override;

  typedef std::unique_ptr<HcalTimeSlew> ReturnType;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const HcalTimeSlewRecord&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  struct M2Parameters {
    float t0, m, tmaximum;
  };
  struct M3Parameters {
    double cap, tspar0, tspar1, tspar2, tspar0_siPM, tspar1_siPM, tspar2_siPM;
  };
  std::vector<M2Parameters> m2parameters_;
  std::vector<M3Parameters> m3parameters_;
};

#endif
