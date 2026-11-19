// -*- C++ -*-
//
// Package:    CondTools/SiPixel
// Class:      SiPixelClusterShapeLimitsReader
//
/**\class SiPixelClusterShapeLimitsReader SiPixelClusterShapeLimitsReader.cc CondTools/SiPixel/plugins/SiPixelClusterShapeLimitsReader.cc
 Description: reads a SiPixelClusterShapeLimits payload from the EventSetup and dumps its content
*/

#include <fstream>
#include <sstream>
#include <string>

#include "CondFormats/DataRecord/interface/SiPixelClusterShapeLimitsRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelClusterShapeLimits.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class SiPixelClusterShapeLimitsReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelClusterShapeLimitsReader(const edm::ParameterSet&);
  ~SiPixelClusterShapeLimitsReader() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::ESGetToken<SiPixelClusterShapeLimits, SiPixelClusterShapeLimitsRcd> limitsToken_;
  edm::ESWatcher<SiPixelClusterShapeLimitsRcd> limitsWatcher_;
  const std::string outputFile_;
  const bool printDebug_;
};

SiPixelClusterShapeLimitsReader::SiPixelClusterShapeLimitsReader(const edm::ParameterSet& iConfig)
    : limitsToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("label")))),
      outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile")),
      printDebug_(iConfig.getUntrackedParameter<bool>("printDebug")) {}

void SiPixelClusterShapeLimitsReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (!limitsWatcher_.check(iSetup))
    return;

  const auto& limits = iSetup.getData(limitsToken_);

  edm::LogInfo("SiPixelClusterShapeLimitsReader")
      << "run " << iEvent.id().run() << ": payload with " << limits.tables().size() << " tables and "
      << limits.rules().size() << " rules";

  // throws if BPix or FPix has no catch-all rule
  limits.checkComplete();

  if (printDebug_) {
    std::ostringstream out;
    limits.printAll(out);
    edm::LogPrint("SiPixelClusterShapeLimitsReader") << out.str();
  }

  if (!outputFile_.empty()) {
    std::ofstream out(outputFile_);
    limits.printAll(out);
  }
}

void SiPixelClusterShapeLimitsReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Reads and dumps a SiPixelClusterShapeLimits payload");
  desc.add<std::string>("label", "");
  desc.addUntracked<std::string>("outputFile", "")->setComment("if not empty, dump the payload content to this file");
  desc.addUntracked<bool>("printDebug", false);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiPixelClusterShapeLimitsReader);
