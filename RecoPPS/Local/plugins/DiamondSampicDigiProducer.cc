// -*- C++ -*-
//
// Package:    SampicDigi/DiamondSampicDigiProducer
// Class:      DiamondSampicDigiProducer
//
/**\class DiamondSampicDigiProducer DiamondSampicDigiProducer.cc SampicDigi/DiamondSampicDigiProducer/plugins/DiamondSampicDigiProducer.cc

 Description: This plugin takes testbeam data as input and converts them to TotemTimingDigi which could be then passed to reco

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Misan (krzysztof.misan@cern.ch)
//         Created:  Sun, 07 Mar 2021 14:42:52 GMT
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "TFile.h"
#include "TTree.h"
#include "TChain.h"

const int MAX_SAMPIC_CHANNELS = 32;
const int SAMPIC_SAMPLES = 64;

//
// class declaration
//

class DiamondSampicDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit DiamondSampicDigiProducer(const edm::ParameterSet&);
  ~DiamondSampicDigiProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  std::vector<std::string> sampicFilesVec;
  std::unordered_map<unsigned int, std::vector<unsigned int>> detid_vs_chid_;
  TChain* inputTree;
  std::stringstream ssLog;
};

//
// constructors and destructor
//

using namespace edm;
using namespace std;

DiamondSampicDigiProducer::DiamondSampicDigiProducer(const edm::ParameterSet& iConfig)
    : sampicFilesVec(iConfig.getParameter<std::vector<std::string>>("sampicFilesVec")) {
  for (const auto& id_map : iConfig.getParameter<std::vector<edm::ParameterSet>>("idsMapping"))
    detid_vs_chid_[id_map.getParameter<unsigned int>("treeChId")] = id_map.getParameter<vector<unsigned int>>("detId");

  inputTree = new TChain("desy");
  for (uint i = 0; i < sampicFilesVec.size(); i++)
    inputTree->Add(sampicFilesVec[i].c_str());

  produces<DetSetVector<TotemTimingDigi>>("TotemTiming");
}

DiamondSampicDigiProducer::~DiamondSampicDigiProducer() { delete inputTree; }

//
// member functions
//

void DiamondSampicDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int eventNum = iEvent.id().event();
  auto digi = std::make_unique<edm::DetSetVector<TotemTimingDigi>>();

  uint num_samples;
  double trigger_time;
  int sample_channel[MAX_SAMPIC_CHANNELS];
  int sample_cellInfoForOrderedData[MAX_SAMPIC_CHANNELS];
  uint sample_timestampA[MAX_SAMPIC_CHANNELS];
  uint sample_timestampB[MAX_SAMPIC_CHANNELS];
  ULong64_t sample_timestampFPGA[MAX_SAMPIC_CHANNELS];
  double sample_amp[MAX_SAMPIC_CHANNELS][SAMPIC_SAMPLES];
  double sample_triggerPosition[MAX_SAMPIC_CHANNELS][SAMPIC_SAMPLES];

  inputTree->SetBranchAddress("num_samples", &num_samples);
  inputTree->SetBranchAddress("trigger_time", &trigger_time);
  inputTree->SetBranchAddress("sample_channel", sample_channel);
  inputTree->SetBranchAddress("sample_timestampFPGA", sample_timestampFPGA);
  inputTree->SetBranchAddress("sample_timestampA", sample_timestampA);
  inputTree->SetBranchAddress("sample_timestampB", sample_timestampB);
  inputTree->SetBranchAddress("sample_cellInfoForOrderedData", sample_cellInfoForOrderedData);
  inputTree->SetBranchAddress("sample_ampl", sample_amp);
  inputTree->SetBranchAddress("sample_triggerPosition", sample_triggerPosition);
  inputTree->GetEntry(eventNum);

  int bunchNumber = ((int)trigger_time / 25) % 3564;
  int orbitNumber = (int)(trigger_time / 88924.45);

  if (num_samples == 0) {
    iEvent.put(std::move(digi), "TotemTiming");
    return;
  }

  for (uint i = 0; i < num_samples; i++) {
    unsigned short ch_id = sample_channel[i];

    //for testbeam data channels<8 don't contain measurements
    if (ch_id < 8)
      continue;

    std::vector<uint8_t> samples;
    unsigned int offsetOfSamples = 0;
    bool search_for_white_cell = true;

    for (int y = 0; y < SAMPIC_SAMPLES; y++) {
      samples.push_back((int)(sample_amp[i][y] / 1.2 * 256));
      if (search_for_white_cell && sample_triggerPosition[i][y] == 1) {
        offsetOfSamples = y;
        search_for_white_cell = false;
      }
    }

    TotemTimingEventInfo eventInfoTmp(0,
                                      trigger_time * 1E9 / 10,  //l1ATimestamp
                                      bunchNumber,
                                      orbitNumber,
                                      eventNum,
                                      1,
                                      1220 / 10,  //l1ALatency
                                      64,
                                      offsetOfSamples,
                                      1);

    TotemTimingDigi digiTmp(2 * 32 + ch_id,
                            sample_timestampFPGA[i],
                            sample_timestampA[i],
                            sample_timestampB[i],
                            sample_cellInfoForOrderedData[i],
                            samples,
                            eventInfoTmp);

    auto vec = detid_vs_chid_.at(ch_id);
    for (auto id : vec) {
      CTPPSDiamondDetId detId(id);
      edm::DetSet<TotemTimingDigi>& digis_for_detid = digi->find_or_insert(detId);
      digis_for_detid.push_back(digiTmp);
    }
  }
  iEvent.put(std::move(digi), "TotemTiming");
}

void DiamondSampicDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<vector<std::string>>("sampicFilesVec")->setComment("path to sampic root data");

  edm::ParameterSetDescription idmap_valid;
  idmap_valid.add<unsigned int>("treeChId", 0)->setComment("HW id as retrieved from tree");
  idmap_valid.add<vector<unsigned int>>("detId")->setComment("mapped CTPPSDiamondDetId's for this channel");

  desc.addVPSet("idsMapping", idmap_valid);

  descriptions.add("DiamondSampicDigiProducer", desc);
}

DEFINE_FWK_MODULE(DiamondSampicDigiProducer);
