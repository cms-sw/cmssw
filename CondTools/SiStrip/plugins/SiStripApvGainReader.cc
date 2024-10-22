// system include files
#include <iostream>
#include <cstdio>
#include <sys/time.h>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"

class SiStripApvGainReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripApvGainReader(const edm::ParameterSet&);
  ~SiStripApvGainReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // initializers list
  const bool printdebug_;
  const std::string formatedOutput_;
  const uint32_t gainType_;
  const edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  edm::Service<TFileService> fs_;
  TTree* tree_ = nullptr;
  int id_ = 0, detId_ = 0, apvId_ = 0;
  double gain_ = 0;
};

using namespace cms;

SiStripApvGainReader::SiStripApvGainReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", true)),
      formatedOutput_(iConfig.getUntrackedParameter<std::string>("outputFile", "")),
      gainType_(iConfig.getUntrackedParameter<uint32_t>("gainType", 1)),
      gainToken_(esConsumes()) {
  usesResource(TFileService::kSharedResource);

  if (fs_.isAvailable()) {
    tree_ = fs_->make<TTree>("Gains", "Gains");

    tree_->Branch("Index", &id_, "Index/I");
    tree_->Branch("DetId", &detId_, "DetId/I");
    tree_->Branch("APVId", &apvId_, "APVId/I");
    tree_->Branch("Gain", &gain_, "Gain/D");
  }
}

void SiStripApvGainReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& stripApvGain = iSetup.getData(gainToken_);
  edm::LogInfo("SiStripApvGainReader") << "[SiStripApvGainReader::analyze] End Reading SiStripApvGain" << std::endl;
  std::vector<uint32_t> detid;
  stripApvGain.getDetIds(detid);
  edm::LogInfo("Number of detids ") << detid.size() << std::endl;

  FILE* pFile = nullptr;
  if (!formatedOutput_.empty())
    pFile = fopen(formatedOutput_.c_str(), "w");
  for (size_t id = 0; id < detid.size(); id++) {
    SiStripApvGain::Range range = stripApvGain.getRange(detid[id], gainType_);
    if (printdebug_) {
      int apv = 0;
      for (int it = 0; it < range.second - range.first; it++) {
        edm::LogInfo("SiStripApvGainReader")
            << "detid " << detid[id] << " \t " << apv++ << " \t " << stripApvGain.getApvGain(it, range) << std::endl;
        id_++;

        if (tree_) {
          detId_ = detid[id];
          apvId_ = apv;
          gain_ = stripApvGain.getApvGain(it, range);
          tree_->Fill();
        }
      }
    }

    if (pFile) {
      fprintf(pFile, "%i ", detid[id]);
      for (int it = 0; it < range.second - range.first; it++) {
        fprintf(pFile, "%f ", stripApvGain.getApvGain(it, range));
      }
      fprintf(pFile, "\n");
    }
  }

  if (pFile)
    fclose(pFile);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripApvGainReader);
