#include <iostream>
#include <memory>

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelFakeGenErrorDBSourceReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFakeGenErrorDBSourceReader(const edm::ParameterSet&);
  ~SiPixelFakeGenErrorDBSourceReader() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::ESWatcher<SiPixelGenErrorDBObjectRcd> SiPixelGenErrorDBObjectWatcher_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrToken_;
};

SiPixelFakeGenErrorDBSourceReader::SiPixelFakeGenErrorDBSourceReader(const edm::ParameterSet& iConfig)
    : genErrToken_(esConsumes()) {}

SiPixelFakeGenErrorDBSourceReader::~SiPixelFakeGenErrorDBSourceReader() = default;

void SiPixelFakeGenErrorDBSourceReader::beginJob() {}

void SiPixelFakeGenErrorDBSourceReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (SiPixelGenErrorDBObjectWatcher_.check(iSetup)) {
    edm::LogPrint("SiPixelFakeGenErrorDBSourceReader") << *&iSetup.getData(genErrToken_) << std::endl;
  }
}

void SiPixelFakeGenErrorDBSourceReader::endJob() {}

DEFINE_FWK_MODULE(SiPixelFakeGenErrorDBSourceReader);
