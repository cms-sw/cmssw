#include <iostream>
#include <memory>

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelFakeTemplateDBSourceReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFakeTemplateDBSourceReader(const edm::ParameterSet&);
  ~SiPixelFakeTemplateDBSourceReader() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::ESWatcher<SiPixelTemplateDBObjectRcd> SiPixelTemplateDBObjectWatcher_;
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectRcd> the1DTemplateToken_;
};

SiPixelFakeTemplateDBSourceReader::SiPixelFakeTemplateDBSourceReader(const edm::ParameterSet& iConfig)
    : the1DTemplateToken_(esConsumes()) {}

SiPixelFakeTemplateDBSourceReader::~SiPixelFakeTemplateDBSourceReader() = default;

void SiPixelFakeTemplateDBSourceReader::beginJob() {}

void SiPixelFakeTemplateDBSourceReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (SiPixelTemplateDBObjectWatcher_.check(iSetup)) {
    edm::LogPrint("SiPixelFakeTemplateDBSourceReader") << *&iSetup.getData(the1DTemplateToken_) << std::endl;
  }
}

void SiPixelFakeTemplateDBSourceReader::endJob() {}

DEFINE_FWK_MODULE(SiPixelFakeTemplateDBSourceReader);
