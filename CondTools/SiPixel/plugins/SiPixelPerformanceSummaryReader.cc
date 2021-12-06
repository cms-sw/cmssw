// system includes
#include <cstdio>
#include <iostream>
#include <sys/time.h>

// user includes
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

using namespace cms;
using namespace std;

namespace cms {
  class SiPixelPerformanceSummaryReader : public edm::one::EDAnalyzer<> {
  public:
    explicit SiPixelPerformanceSummaryReader(const edm::ParameterSet&);
    ~SiPixelPerformanceSummaryReader() override;

    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    const edm::ESGetToken<SiPixelPerformanceSummary, SiPixelPerformanceSummaryRcd> perfSummaryToken_;
    const bool printdebug_;
  };
}  // namespace cms

SiPixelPerformanceSummaryReader::SiPixelPerformanceSummaryReader(const edm::ParameterSet& iConfig)
    : perfSummaryToken_(esConsumes()), printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {}

SiPixelPerformanceSummaryReader::~SiPixelPerformanceSummaryReader() = default;

void SiPixelPerformanceSummaryReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "start reading SiPixelPerformanceSummary" << endl;
  const SiPixelPerformanceSummary* SiPixelPerformanceSummary_ = &iSetup.getData(perfSummaryToken_);
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "end reading SiPixelPerformanceSummary" << endl;

  SiPixelPerformanceSummary_->print();
  vector<uint32_t> allDetIds = SiPixelPerformanceSummary_->getAllDetIds();
  if (!allDetIds.empty())
    SiPixelPerformanceSummary_->print(allDetIds[0]);
  SiPixelPerformanceSummary_->printAll();
}
DEFINE_FWK_MODULE(SiPixelPerformanceSummaryReader);
