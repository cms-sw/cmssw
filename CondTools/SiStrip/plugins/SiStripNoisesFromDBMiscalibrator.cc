// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripNoisesFromDBMiscalibrator
//
/**\class SiStripNoisesFromDBMiscalibrator SiStripNoisesFromDBMiscalibrator.cc CondTools/SiStrip/plugins/SiStripNoisesFromDBMiscalibrator.cc

 Description: Class to miscalibrate a SiStrip Noise payload from Database

 Implementation:
     Read a SiStrip Noise payload from DB (either central DB or sqlite file) and apply a miscalibration (either an offset / gaussian smearing or both)
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
#include <fstream>

// user include files
#include "CLHEP/Random/RandGauss.h"
#include "CondTools/SiStrip/interface/SiStripMiscalibrateHelper.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
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

class SiStripNoisesFromDBMiscalibrator : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripNoisesFromDBMiscalibrator(const edm::ParameterSet&);
  ~SiStripNoisesFromDBMiscalibrator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::unique_ptr<SiStripNoises> getNewObject(const std::map<std::pair<uint32_t, int>, float>& theMap);
  std::unique_ptr<SiStripNoises> getNewObject_withDefaults(const std::map<std::pair<uint32_t, int>, float>& theMap,
                                                           const float theDefault);
  void endJob() override;

  // ----------member data ---------------------------
  const bool m_fillDefaults;
  const bool m_saveMaps;
  const std::vector<edm::ParameterSet> m_parameters;
  edm::FileInPath fp_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> m_noiseToken;

  std::unique_ptr<TrackerMap> scale_map;
  std::unique_ptr<TrackerMap> smear_map;
  std::unique_ptr<TrackerMap> ratio_map;
  std::unique_ptr<TrackerMap> old_payload_map;
  std::unique_ptr<TrackerMap> new_payload_map;
  std::unique_ptr<TrackerMap> missing_map;
};

//
// constructors and destructor
//
SiStripNoisesFromDBMiscalibrator::SiStripNoisesFromDBMiscalibrator(const edm::ParameterSet& iConfig)
    : m_fillDefaults{iConfig.getUntrackedParameter<bool>("fillDefaults", false)},
      m_saveMaps{iConfig.getUntrackedParameter<bool>("saveMaps", true)},
      m_parameters{iConfig.getParameter<std::vector<edm::ParameterSet> >("params")},
      fp_{iConfig.getUntrackedParameter<edm::FileInPath>(
          "file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))},
      m_tTopoToken(esConsumes()),
      m_noiseToken(esConsumes()) {
  //now do what ever initialization is needed

  scale_map = std::make_unique<TrackerMap>("scale");
  scale_map->setTitle("Tracker Map of Scale factor averaged by module");
  scale_map->setPalette(1);

  smear_map = std::make_unique<TrackerMap>("smear");
  smear_map->setTitle("Tracker Map of Smear factor averaged by module");
  smear_map->setPalette(1);

  old_payload_map = std::make_unique<TrackerMap>("old_payload");
  old_payload_map->setTitle("Tracker Map of Starting Noise Payload averaged by module");
  old_payload_map->setPalette(1);

  new_payload_map = std::make_unique<TrackerMap>("new_payload");
  new_payload_map->setTitle("Tracker Map of Modified Noise Payload averaged by module");
  new_payload_map->setPalette(1);

  ratio_map = std::make_unique<TrackerMap>("ratio");
  ratio_map->setTitle("Tracker Map of Average by module of the payload ratio (new/old)");
  ratio_map->setPalette(1);

  if (m_fillDefaults) {
    missing_map = std::make_unique<TrackerMap>("uncabled");
    missing_map->setTitle("Tracker Map of uncabled modules");
    missing_map->setPalette(1);
  }
}

SiStripNoisesFromDBMiscalibrator::~SiStripNoisesFromDBMiscalibrator() {}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripNoisesFromDBMiscalibrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto tTopo = &iSetup.getData(m_tTopoToken);

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

  const auto& stripNoises = iSetup.getData(m_noiseToken);

  std::map<std::pair<uint32_t, int>, float> theMap, oldPayloadMap;

  std::vector<uint32_t> detid;
  stripNoises.getDetIds(detid);
  for (const auto& d : detid) {
    SiStripNoises::Range range = stripNoises.getRange(d);

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

    int nStrips = 0;
    for (int it = 0; it < (range.second - range.first) * 8 / 9; ++it) {
      auto noise = stripNoises.getNoise(it, range);
      std::pair<uint32_t, int> index = std::make_pair(d, nStrips);

      oldPayloadMap[index] = noise;

      if (params.m_doScale) {
        noise *= params.m_scaleFactor;
      }

      if (params.m_doSmear) {
        float smearedNoise = CLHEP::RandGauss::shoot(noise, params.m_smearFactor);
        noise = smearedNoise;
      }

      theMap[index] = noise;

      nStrips += 1;

    }  // loop over APVs
  }    // loop over DetIds

  std::unique_ptr<SiStripNoises> theSiStripNoises;
  if (!m_fillDefaults) {
    theSiStripNoises = this->getNewObject(theMap);
  } else {
    theSiStripNoises = this->getNewObject_withDefaults(theMap, -1.);
  }

  // make the payload ratio map
  uint32_t cachedId(0);
  SiStripMiscalibrate::Entry noise_ratio;
  SiStripMiscalibrate::Entry o_noise;
  SiStripMiscalibrate::Entry n_noise;
  for (const auto& element : theMap) {
    uint32_t DetId = element.first.first;
    int nstrip = element.first.second;
    float new_noise = element.second;
    float old_noise = oldPayloadMap[std::make_pair(DetId, nstrip)];

    // flush the counters
    if (cachedId != 0 && DetId != cachedId) {
      ratio_map->fill(cachedId, noise_ratio.mean());
      old_payload_map->fill(cachedId, o_noise.mean());
      new_payload_map->fill(cachedId, n_noise.mean());

      //auto test = new_payload_map.get()->smoduleMap;

      noise_ratio.reset();
      o_noise.reset();
      n_noise.reset();
    }

    cachedId = DetId;
    noise_ratio.add(new_noise / old_noise);
    o_noise.add(old_noise);
    n_noise.add(new_noise);
  }

  // write out the SiStripNoises record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOne(theSiStripNoises.get(), poolDbService->currentTime(), "SiStripNoisesRcd");
  else
    throw std::runtime_error("PoolDBService required.");
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripNoisesFromDBMiscalibrator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripNoisesFromDBMiscalibrator::endJob() {
  if (m_saveMaps) {
    scale_map->save(true, 0, 0, "noise_scale_map.pdf");
    scale_map->save(true, 0, 0, "noise_scale_map.png");

    smear_map->save(true, 0, 0, "noise_smear_map.pdf");
    smear_map->save(true, 0, 0, "noise_smear_map.png");

    ratio_map->save(true, 0, 0, "noise_ratio_map.pdf");
    ratio_map->save(true, 0, 0, "noise_ratio_map.png");

    auto range = SiStripMiscalibrate::getTruncatedRange(old_payload_map.get());

    old_payload_map->save(true, range.first, range.second, "starting_noise_payload_map.pdf");
    old_payload_map->save(true, range.first, range.second, "starting_noise_payload_map.png");

    range = SiStripMiscalibrate::getTruncatedRange(new_payload_map.get());

    new_payload_map->save(true, range.first, range.second, "new_noise_payload_map.pdf");
    new_payload_map->save(true, range.first, range.second, "new_noise_payload_map.png");

    if (m_fillDefaults) {
      missing_map->save(true, 0, 0, "missing_map.pdf");
      missing_map->save(true, 0, 0, "missing_map.png");
    }
  }
}

//********************************************************************************//
std::unique_ptr<SiStripNoises> SiStripNoisesFromDBMiscalibrator::getNewObject_withDefaults(
    const std::map<std::pair<uint32_t, int>, float>& theMap, const float theDefault) {
  std::unique_ptr<SiStripNoises> obj = std::make_unique<SiStripNoises>();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>& DetInfos = reader.getAllData();

  std::vector<uint32_t> missingDetIds;

  for (std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>::const_iterator it = DetInfos.begin();
       it != DetInfos.end();
       it++) {
    //Generate Noise for det detid
    bool isMissing(false);
    SiStripNoises::InputVector theSiStripVector;
    for (int t_strip = 0; t_strip < 128 * it->second.nApvs; ++t_strip) {
      std::pair<uint32_t, int> index = std::make_pair(it->first, t_strip);

      if (theMap.find(index) == theMap.end()) {
        LogDebug("SiStripNoisesFromDBMiscalibrator") << "detid " << it->first << " \t"
                                                     << " strip " << t_strip << " \t"
                                                     << " not found" << std::endl;

        isMissing = true;
        obj->setData(theDefault, theSiStripVector);

      } else {
        float noise = theMap.at(index);
        obj->setData(noise, theSiStripVector);
      }
    }

    if (isMissing)
      missingDetIds.push_back(it->first);

    if (!obj->put(it->first, theSiStripVector)) {
      edm::LogError("SiStripNoisesFromDBMiscalibrator")
          << "[SiStripNoisesFromDBMiscalibrator::analyze] detid already exists" << std::endl;
    }
  }

  if (!missingDetIds.empty()) {
    // open output file
    std::stringstream name;
    name << "missing_modules.txt";
    std::ofstream* ofile = new std::ofstream(name.str(), std::ofstream::trunc);
    if (!ofile->is_open())
      throw "cannot open output file!";
    for (const auto& missing : missingDetIds) {
      edm::LogVerbatim("SiStripNoisesFromDBMiscalibrator") << missing << "  " << 1 << std::endl;
      (*ofile) << missing << " " << 1 << std::endl;
      missing_map->fill(missing, 1);
    }

    ofile->close();
    delete ofile;
  }

  return obj;
}

//********************************************************************************//
std::unique_ptr<SiStripNoises> SiStripNoisesFromDBMiscalibrator::getNewObject(
    const std::map<std::pair<uint32_t, int>, float>& theMap) {
  std::unique_ptr<SiStripNoises> obj = std::make_unique<SiStripNoises>();

  uint32_t PreviousDetId = 0;
  SiStripNoises::InputVector theSiStripVector;
  for (const auto& element : theMap) {
    uint32_t DetId = element.first.first;
    float noise = element.second;

    if (DetId != PreviousDetId) {
      if (!theSiStripVector.empty()) {
        if (!obj->put(PreviousDetId, theSiStripVector)) {
          edm::LogError("SiStripNoisesFromDBMiscalibrator")
              << "[SiStripNoisesFromDBMiscalibrator::analyze] detid already exists" << std::endl;
        }
      }

      theSiStripVector.clear();
      PreviousDetId = DetId;
    }
    obj->setData(noise, theSiStripVector);
  }
  return obj;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripNoisesFromDBMiscalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      "Creates rescaled / smeared SiStrip Noise payload."
      "PoolDBOutputService must be set up for 'SiSiStripNoisesRcd'.");

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

  desc.addUntracked<bool>("fillDefaults", false);
  desc.addUntracked<bool>("saveMaps", true);

  descriptions.add("scaleAndSmearSiStripNoises", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripNoisesFromDBMiscalibrator);
