#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class SiStripHitEfficiencyHarvester : public DQMEDHarvester {
public:
  explicit SiStripHitEfficiencyHarvester(const edm::ParameterSet&);
  ~SiStripHitEfficiencyHarvester() override = default;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
private:
};

SiStripHitEfficiencyHarvester::SiStripHitEfficiencyHarvester(const edm::ParameterSet&) {}

void SiStripHitEfficiencyHarvester::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) {

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyHarvester);
