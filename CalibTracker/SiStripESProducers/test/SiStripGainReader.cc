// system include files
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class decleration
//
class SiStripGainReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripGainReader(const edm::ParameterSet&);
  ~SiStripGainReader() = default;

  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  bool printdebug_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken;
};

SiStripGainReader::SiStripGainReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)), gainToken(esConsumes()) {}

void SiStripGainReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& SiStripGain_ = &iSetup.getData(gainToken);
  edm::LogInfo("SiStripGainReader") << "[SiStripGainReader::analyze] End Reading SiStripGain";

  std::vector<uint32_t> detid;
  SiStripGain_->getDetIds(detid);

  SiStripApvGain::Range range = SiStripGain_->getRange(detid[0]);

  edm::LogInfo("SiStripApvGainReader") << "Number of detids: " << detid.size();
  int apv = 0;
  edm::LogInfo("SiStripApvGainReader") << " First det gain values";
  for (int it = 0; it < range.second - range.first; it++) {
    edm::LogInfo("SiStripApvGainReader") << "detid " << detid[0] << " \t"
                                         << " apv " << apv++ << " \t" << SiStripGain_->getApvGain(it, range) << " \t";
  }

  if (printdebug_) {
    for (size_t id = 0; id < detid.size(); id++) {
      SiStripApvGain::Range range = SiStripGain_->getRange(detid[id]);
      apv = 0;

      for (int it = 0; it < range.second - range.first; it++) {
        edm::LogInfo("SiStripGainReader") << "detid " << detid[id] << " \t"
                                          << " apv " << apv++ << " \t" << SiStripGain_->getApvGain(it, range) << " \t";
      }
    }
  }
}

DEFINE_FWK_MODULE(SiStripGainReader);
