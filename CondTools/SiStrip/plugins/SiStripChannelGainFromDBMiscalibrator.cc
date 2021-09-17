// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripChannelGainFromDBMiscalibrator
//
/**\class SiStripChannelGainFromDBMiscalibrator SiStripChannelGainFromDBMiscalibrator.cc CondTools/SiStrip/plugins/SiStripChannelGainFromDBMiscalibrator.cc

 Description: Class to miscalibrate a SiStrip Channel Gain payload from Database

 Implementation:
     Read a SiStrip Channel Gain payload from DB (either central DB or sqlite file) and apply a miscalibration (either an offset / gaussian smearing or both)
     returns a local sqlite file with the same since of the original payload
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 03 Oct 2017 12:57:34 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "CLHEP/Random/RandGauss.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "CondTools/SiStrip/interface/SiStripMiscalibrateHelper.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
//
// class declaration
//

class SiStripChannelGainFromDBMiscalibrator : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripChannelGainFromDBMiscalibrator(const edm::ParameterSet&);
  ~SiStripChannelGainFromDBMiscalibrator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::unique_ptr<SiStripApvGain> getNewObject(const std::map<std::pair<uint32_t, int>, float>& theMap);
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_Record;
  const uint32_t m_gainType;
  const bool m_saveMaps;
  const std::vector<edm::ParameterSet> m_parameters;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  std::unique_ptr<TrackerMap> scale_map;
  std::unique_ptr<TrackerMap> smear_map;
  std::unique_ptr<TrackerMap> ratio_map;
  std::unique_ptr<TrackerMap> old_payload_map;
  std::unique_ptr<TrackerMap> new_payload_map;
};

//
// constructors and destructor
//
SiStripChannelGainFromDBMiscalibrator::SiStripChannelGainFromDBMiscalibrator(const edm::ParameterSet& iConfig)
    : m_Record{iConfig.getUntrackedParameter<std::string>("record", "SiStripApvGainRcd")},
      m_gainType{iConfig.getUntrackedParameter<uint32_t>("gainType", 1)},
      m_saveMaps{iConfig.getUntrackedParameter<bool>("saveMaps", true)},
      m_parameters{iConfig.getParameter<std::vector<edm::ParameterSet> >("params")},
      gainToken_(esConsumes()),
      tTopoToken_(esConsumes()) {
  //now do what ever initialization is needed

  std::string ss_gain = (m_gainType > 0) ? "G2" : "G1";

  scale_map = std::make_unique<TrackerMap>("scale");
  scale_map->setTitle("Scale factor averaged by module");
  scale_map->setPalette(1);

  smear_map = std::make_unique<TrackerMap>("smear");
  smear_map->setTitle("Smear factor averaged by module");
  smear_map->setPalette(1);

  ratio_map = std::make_unique<TrackerMap>("ratio");
  ratio_map->setTitle("Average by module of the " + ss_gain + " Gain payload ratio (new/old)");
  ratio_map->setPalette(1);

  new_payload_map = std::make_unique<TrackerMap>("new_payload");
  new_payload_map->setTitle("Tracker Map of Modified " + ss_gain + " Gain payload averaged by module");
  new_payload_map->setPalette(1);

  old_payload_map = std::make_unique<TrackerMap>("old_payload");
  old_payload_map->setTitle("Tracker Map of Starting " + ss_gain + " Gain Payload averaged by module");
  old_payload_map->setPalette(1);
}

SiStripChannelGainFromDBMiscalibrator::~SiStripChannelGainFromDBMiscalibrator() {}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripChannelGainFromDBMiscalibrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto* const tTopo = &iSetup.getData(tTopoToken_);

  std::vector<std::string> partitions;

  // fill the list of partitions
  for (auto& thePSet : m_parameters) {
    const std::string partition(thePSet.getParameter<std::string>("partition"));
    // only if it is not yet in the list
    if (std::find(partitions.begin(), partitions.end(), partition) == partitions.end()) {
      partitions.push_back(partition);
    }
  }

  std::map<sistripsummary::TrackerRegion, SiStripMiscalibrate::Smearings> mapOfSmearings;

  for (auto& thePSet : m_parameters) {
    const std::string partition(thePSet.getParameter<std::string>("partition"));
    sistripsummary::TrackerRegion region = SiStripMiscalibrate::getRegionFromString(partition);

    bool m_doScale(thePSet.getParameter<bool>("doScale"));
    bool m_doSmear(thePSet.getParameter<bool>("doSmear"));
    double m_scaleFactor(thePSet.getParameter<double>("scaleFactor"));
    double m_smearFactor(thePSet.getParameter<double>("smearFactor"));

    SiStripMiscalibrate::Smearings params = SiStripMiscalibrate::Smearings();
    params.setSmearing(m_doScale, m_doSmear, m_scaleFactor, m_smearFactor);
    mapOfSmearings[region] = params;
  }

  const auto& apvGain = iSetup.getData(gainToken_);

  std::map<std::pair<uint32_t, int>, float> theMap, oldPayloadMap;

  std::vector<uint32_t> detid;
  apvGain.getDetIds(detid);
  for (const auto& d : detid) {
    SiStripApvGain::Range range = apvGain.getRange(d, m_gainType);
    float nAPV = 0;

    auto regions = SiStripMiscalibrate::getRegionsFromDetId(tTopo, d);

    // sort by largest to smallest
    std::sort(regions.rbegin(), regions.rend());

    SiStripMiscalibrate::Smearings params = SiStripMiscalibrate::Smearings();

    for (unsigned int j = 0; j < regions.size(); j++) {
      bool checkRegion = (mapOfSmearings.count(regions[j]) != 0);

      if (!checkRegion) {
        // if the subdetector is not in the list and there's no indication for the whole tracker, just use the default
        // i.e. no change
        continue;
      } else {
        params = mapOfSmearings[regions[j]];
        break;
      }
    }

    scale_map->fill(d, params.m_scaleFactor);
    smear_map->fill(d, params.m_smearFactor);

    for (int it = 0; it < range.second - range.first; it++) {
      nAPV += 1;
      float Gain = apvGain.getApvGain(it, range);
      std::pair<uint32_t, int> index = std::make_pair(d, nAPV);

      oldPayloadMap[index] = Gain;

      if (params.m_doScale) {
        Gain *= params.m_scaleFactor;
      }

      if (params.m_doSmear) {
        float smearedGain = CLHEP::RandGauss::shoot(Gain, params.m_smearFactor);
        Gain = smearedGain;
      }

      theMap[index] = Gain;

    }  // loop over APVs
  }    // loop over DetIds

  std::unique_ptr<SiStripApvGain> theAPVGains = this->getNewObject(theMap);

  // make the payload ratio map
  uint32_t cachedId(0);
  SiStripMiscalibrate::Entry gain_ratio;
  SiStripMiscalibrate::Entry o_gain;
  SiStripMiscalibrate::Entry n_gain;
  for (const auto& element : theMap) {
    uint32_t DetId = element.first.first;
    int nAPV = element.first.second;
    float new_gain = element.second;
    float old_gain = oldPayloadMap[std::make_pair(DetId, nAPV)];

    // flush the counters
    if (cachedId != 0 && DetId != cachedId) {
      ratio_map->fill(cachedId, gain_ratio.mean());
      old_payload_map->fill(cachedId, o_gain.mean());
      new_payload_map->fill(cachedId, n_gain.mean());

      //auto test = new_payload_map.get()->smoduleMap;

      gain_ratio.reset();
      o_gain.reset();
      n_gain.reset();
    }

    cachedId = DetId;
    gain_ratio.add(new_gain / old_gain);
    o_gain.add(old_gain);
    n_gain.add(new_gain);
  }

  // write out the APVGains record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOne(theAPVGains.get(), poolDbService->currentTime(), m_Record);
  else
    throw std::runtime_error("PoolDBService required.");
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripChannelGainFromDBMiscalibrator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripChannelGainFromDBMiscalibrator::endJob() {
  if (m_saveMaps) {
    std::string ss_gain = (m_gainType > 0) ? "G2" : "G1";

    scale_map->save(true, 0, 0, ss_gain + "_gain_scale_map.pdf");
    scale_map->save(true, 0, 0, ss_gain + "_gain_scale_map.png");

    smear_map->save(true, 0, 0, ss_gain + "_gain_smear_map.pdf");
    smear_map->save(true, 0, 0, ss_gain + "_gain_smear_map.png");

    ratio_map->save(true, 0, 0, ss_gain + "_gain_ratio_map.pdf");
    ratio_map->save(true, 0, 0, ss_gain + "_gain_ratio_map.png");

    auto range = SiStripMiscalibrate::getTruncatedRange(old_payload_map.get());

    old_payload_map->save(true, range.first, range.second, "starting_" + ss_gain + "_gain_payload_map.pdf");
    old_payload_map->save(true, range.first, range.second, "starting_" + ss_gain + "_gain_payload_map.png");

    range = SiStripMiscalibrate::getTruncatedRange(new_payload_map.get());

    new_payload_map->save(true, range.first, range.second, "new_" + ss_gain + "_gain_payload_map.pdf");
    new_payload_map->save(true, range.first, range.second, "new_" + ss_gain + "_gain_payload_map.png");
  }
}

//********************************************************************************//
std::unique_ptr<SiStripApvGain> SiStripChannelGainFromDBMiscalibrator::getNewObject(
    const std::map<std::pair<uint32_t, int>, float>& theMap) {
  std::unique_ptr<SiStripApvGain> obj = std::make_unique<SiStripApvGain>();

  std::vector<float> theSiStripVector;
  uint32_t PreviousDetId = 0;
  for (const auto& element : theMap) {
    uint32_t DetId = element.first.first;
    if (DetId != PreviousDetId) {
      if (!theSiStripVector.empty()) {
        SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
        if (!obj->put(PreviousDetId, range))
          printf("Bug to put detId = %i\n", PreviousDetId);
      }
      theSiStripVector.clear();
      PreviousDetId = DetId;
    }
    theSiStripVector.push_back(element.second);

    edm::LogInfo("SiStripChannelGainFromDBMiscalibrator")
        << " DetId: " << DetId << " APV:   " << element.first.second << " Gain:  " << element.second << std::endl;
  }

  if (!theSiStripVector.empty()) {
    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(PreviousDetId, range))
      printf("Bug to put detId = %i\n", PreviousDetId);
  }

  return obj;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripChannelGainFromDBMiscalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      "Creates rescaled / smeared SiStrip Gain payload. Can be used for both G1 and G2."
      "PoolDBOutputService must be set up for 'SiStripApvGainRcd'.");

  edm::ParameterSetDescription descScaler;
  descScaler.setComment(
      "ParameterSet specifying the Strip tracker partition to be scaled / smeared "
      "by a given factor.");

  descScaler.add<std::string>("partition", "Tracker");
  descScaler.add<bool>("doScale", true);
  descScaler.add<bool>("doSmear", true);
  descScaler.add<double>("scaleFactor", 1.0);
  descScaler.add<double>("smearFactor", 1.0);
  desc.addVPSet("params", descScaler, std::vector<edm::ParameterSet>(1));

  desc.addUntracked<std::string>("record", "SiStripApvGainRcd");
  desc.addUntracked<unsigned int>("gainType", 1);
  desc.addUntracked<bool>("saveMaps", true);

  descriptions.add("scaleAndSmearSiStripGains", desc);
}

/*--------------------------------------------------------------------*/

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripChannelGainFromDBMiscalibrator);
