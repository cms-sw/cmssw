#ifndef CalibCalorimetry_HcalPlugins_HBHEDarkeningEP_H
#define CalibCalorimetry_HcalPlugins_HBHEDarkeningEP_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HBHEDarkeningEP : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HBHEDarkeningEP(const edm::ParameterSet&);
  ~HBHEDarkeningEP() override;

  typedef std::unique_ptr<HBHEDarkening> ReturnType;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const HBHEDarkeningRecord&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  struct Dosemap {
    edm::FileInPath fp;
    int file_energy;
  };
  std::vector<Dosemap> dosemaps_;
  std::vector<HBHEDarkening::LumiYear> years_;
  const double drdA_;
  const double drdB_;
  const int ieta_shift_;
};

#endif
