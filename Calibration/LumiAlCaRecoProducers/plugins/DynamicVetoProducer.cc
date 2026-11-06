/**_________________________________________________________________

Author: Braden Kronheim and Peter Major

Algo description:
https://indico.cern.ch/event/1358674/#34-pcc-active-masking
________________________________________________________________**/
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <TROOT.h>
#include <utility>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TLine.h"
#include "TMath.h"

#include "Calibration/LumiAlCaRecoProducers/plugins/DQMOneEDProducer.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/PccVetoListRcd.h"
#include "CondFormats/Luminosity/interface/PccVetoList.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/PccVetoListTransient.h"
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

enum class TrackerRegion {
  Any,
  BPix_1,
  BPix_2,
  BPix_3,
  nBPix_4,
  Epix_1_ring1,
  Epix_2_ring1,
  Epix_3_ring1,
  Epix_1_ring2,
  Epix_2_ring2,
  Epix_3_ring2,
};

// class DynamicVetoProducer : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
class DynamicVetoProducer : public DQMOneEDProducer<edm::one::WatchLuminosityBlocks> {
public:
  explicit DynamicVetoProducer(const edm::ParameterSet&);
  ~DynamicVetoProducer() override;

private:
  // values to set up in the config
  edm::EDGetTokenT<reco::PixelClusterCounts> pccToken_;
  edm::EDPutTokenT<PccVetoListTransient> putToken_;

  std::vector<int> baseVeto_;
  bool saveBaseVeto_;
  std::vector<int> moduleListRing1_;
  int lumisectionCountMin_;
  double stdMultiplyer1_;
  double stdMultiplyer2_;
  double stdMultiplyer3_;
  double fractionThreshold2_;
  int filterLevel_;

  std::vector<int> fractionalResponse_modID;
  std::vector<double> fractionalResponse_value;

  bool savePlots_;

  bool coutOn_;

  bool saveCSVFile_;
  std::string csvOutLabel_;
  mutable std::mutex fileLock_;

  // working containers
  //// for round 1
  std::vector<int> additionalVeto1_;
  std::map<TrackerRegion, std::map<int, double> > region2moduleID2countRatio;
  int lumisectionCount_ = 0;
  // std::map<TrackerRegion, double > region2center;
  // std::map<TrackerRegion, double > region2std;
  // std::map<TrackerRegion, int >   region2badModuleCount;

  //// for round 2
  std::vector<int> additionalVeto2_;
  std::map<TrackerRegion, std::map<int, std::vector<unsigned int> > > region2moduleID2LS2counts;

  //// for round 3
  std::vector<int> additionalVeto3_;
  std::map<int, double> fractionalResponseMap;

  // TFile* histoFile;
  // std::map<TrackerRegion, TH1F* > region2countHistogram;

  // geometry
  TrackerRegion getTrackerRegion1(unsigned int mId);
  TrackerRegion getTrackerRegion2(unsigned int mId);

  // math
  double getMean(const std::map<int, unsigned int>& moduleID2value);
  double getStd(const std::map<int, unsigned int>& moduleID2value, const double mean);
  double interp(double v0, double v1, double t) { return (1 - t) * v0 + t * v1; }
  std::vector<double> getQuantile(const std::vector<double>& inData, const std::vector<double>& probs);
  std::vector<double> getQuantile(const std::map<int, double>& inData, const std::vector<double>& probs);

  // select bad modules
  int addBadModules(const std::map<int, double>& moduleID2value,
                    const double mean,
                    const double distance,
                    std::vector<int>& badModules);

  // actions
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  // void dqmEndRun(const edm::Run & runSeg, const edm::EventSetup& iSetup);
  void dqmEndRun(edm::Run& runSeg, const edm::EventSetup& iSetup) override;
  void endJob() final;

  // must have
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& context) override {}

  //
  void makePlot(std::string name,
                std::map<int, double> moduleID2value,
                double center,
                double std,
                double distance,
                int badModuleCount);
  void resetContainers();

  // DB Service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
};

//--------------------------------------------------------------------------------------------------
DynamicVetoProducer::DynamicVetoProducer(const edm::ParameterSet& iConfig)
    : pccToken_(consumes<reco::PixelClusterCounts, edm::InLumi>(
          edm::InputTag(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters")
                            .getParameter<std::string>("inputPccLabel"),
                        iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters")
                            .getParameter<std::string>("prodInst")))) {
  auto pset = iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters");

  putToken_ = produces<PccVetoListTransient, edm::Transition::EndRun>(
      pset.getUntrackedParameter<std::string>("outputProductName", "alcaPccVetoList"));

  baseVeto_ = pset.getParameter<std::vector<int> >("BaseVeto");
  saveBaseVeto_ = pset.getParameter<bool>("SaveBaseVeto");
  fractionalResponse_modID = pset.getParameter<std::vector<int> >("FractionalResponse_modID");
  fractionalResponse_value = pset.getParameter<std::vector<double> >("FractionalResponse_value");
  moduleListRing1_ = pset.getUntrackedParameter<std::vector<int> >("ModuleListRing1", {});
  lumisectionCountMin_ = pset.getUntrackedParameter<int>("MinimumLSCount", 10);
  stdMultiplyer1_ = pset.getParameter<double>("StdMultiplyier1");
  stdMultiplyer2_ = pset.getParameter<double>("StdMultiplyier2");
  fractionThreshold2_ = pset.getParameter<double>("FractionThreshold2");
  stdMultiplyer3_ = pset.getParameter<double>("StdMultiplyier3");
  filterLevel_ = pset.getParameter<int>("FilterLevel");
  savePlots_ = pset.getUntrackedParameter<bool>("SavePlots", false);
  coutOn_ = pset.getUntrackedParameter<bool>("CoutOn", false);
  saveCSVFile_ = pset.getUntrackedParameter<bool>("SaveCSVFile", false);
  csvOutLabel_ = pset.getUntrackedParameter<std::string>("CsvFileName", std::string("dynamicVeto.csv"));

  if (fractionalResponse_modID.size() != fractionalResponse_value.size())
    throw std::runtime_error("Fractional response modID and value lists are not equally long in DynamicVetoProducer.");
  for (size_t i = 0; i < fractionalResponse_modID.size(); i++)
    fractionalResponseMap[fractionalResponse_modID.at(i)] = fractionalResponse_value.at(i);
}

//--------------------------------------------------------------------------------------------------
DynamicVetoProducer::~DynamicVetoProducer() {}

//--------------------------------------------------------------------------------------------------
double DynamicVetoProducer::getMean(const std::map<int, unsigned int>& moduleID2value) {
  double sum = 0;
  for (const auto& [key, value] : moduleID2value) {
    sum += value;
  }
  return sum / moduleID2value.size();
}

double DynamicVetoProducer::getStd(const std::map<int, unsigned int>& moduleID2value, const double mean) {
  double sum2 = 0;
  for (const auto& [key, value] : moduleID2value) {
    sum2 += std::pow(value, 2);
  }
  return std::pow(sum2 / moduleID2value.size() - mean * mean, 0.5);
}

std::vector<double> DynamicVetoProducer::getQuantile(const std::vector<double>& inData,
                                                     const std::vector<double>& probs) {
  if (inData.empty() || probs.empty())
    return std::vector<double>();
  if (1 == inData.size())
    return std::vector<double>(probs.size(), inData[0]);

  std::vector<double> data = inData;
  std::sort(data.begin(), data.end());

  std::vector<double> quantiles;
  for (size_t i = 0; i < probs.size(); ++i) {
    double target = interp(-0.5, data.size() - 0.5, probs[i]);

    size_t left = std::max(int64_t(std::floor(target)), int64_t(0));
    size_t right = std::min(int64_t(std::ceil(target)), int64_t(data.size() - 1));

    double datLeft = data.at(left);
    double datRight = data.at(right);

    quantiles.push_back(interp(datLeft, datRight, target - left));
  }

  return quantiles;
}

std::vector<double> DynamicVetoProducer::getQuantile(const std::map<int, double>& inData,
                                                     const std::vector<double>& probs) {
  if (inData.empty() || probs.empty())
    return std::vector<double>();
  if (1 == inData.size())
    return std::vector<double>(probs.size(), inData.begin()->second);

  std::vector<double> data;
  data.reserve(inData.size());
  for (const auto& [key, value] : inData)
    data.push_back(value);
  std::sort(data.begin(), data.end());

  std::vector<double> quantiles;
  for (size_t i = 0; i < probs.size(); ++i) {
    double target = interp(-0.5, data.size() - 0.5, probs[i]);

    size_t left = std::max(int64_t(std::floor(target)), int64_t(0));
    size_t right = std::min(int64_t(std::ceil(target)), int64_t(data.size() - 1));

    double datLeft = data.at(left);
    double datRight = data.at(right);

    quantiles.push_back(interp(datLeft, datRight, target - left));
  }

  return quantiles;
}

//--------------------------------------------------------------------------------------------------
TrackerRegion DynamicVetoProducer::getTrackerRegion1(unsigned int mId) {
  //https://gitlab.cern.ch/cms-sw/cmssw/tree/7eb5fd3cd39b94ea5618c5c177817c10c7675428/Geometry/TrackerNumberingBuilder
  // I tested all BPix modules by bitwis ANDing ring 1 and ring 2 moduleIDs separately. The results are the same, therefore must use LUT.
  // ring 1 has fewer elements, so that is used

  int region = 0;
  unsigned int subdetectorId = ((mId >> 25) & 0x7);
  if (subdetectorId == 1) {
    region = ((mId >> 20) & 7);
  } else if (subdetectorId == 2) {
    region = ((mId >> 18) & 7);
    region += 4;
    if (std::find(moduleListRing1_.begin(), moduleListRing1_.end(), mId) == moduleListRing1_.end())
      region += 3;
  } else
    throw std::runtime_error("SubdetectorId not found in DynamicVetoProducer::getTrackerRegion.");
  return static_cast<TrackerRegion>(region);
}

TrackerRegion DynamicVetoProducer::getTrackerRegion2(unsigned int mId) {
  // https://gitlab.cern.ch/cms-sw/cmssw/tree/7eb5fd3cd39b94ea5618c5c177817c10c7675428/Geometry/TrackerNumberingBuilder
  // This first snippet allows to get the blade of a given detId:
  // https://github.com/cms-sw/cmssw/blob/CMSSW_15_0_0/DataFormats/SiPixelDetId/interface/PXFDetId.h#L38
  // And this second one allows to get the ring from a blade:
  // https://github.com/cms-sw/cmssw/blob/CMSSW_15_0_0/DataFormats/TrackerCommon/src/PixelEndcapName.cc#L98-L139

  int region = 0;
  unsigned int subdetectorId = ((mId >> 25) & 0x7);
  if (subdetectorId == 1) {
    region = ((mId >> 20) & 7);
  } else if (subdetectorId == 2) {
    region = ((mId >> 18) & 7);
    region += 4;

    unsigned int tmpBlade = ((mId >> 10) & 0x3F);

    int ring = 0;  // ring number , according to radius, 1-lower, 2- higher
    if (tmpBlade <= 22)
      ring = 1;
    else if (tmpBlade >= 23 && tmpBlade <= 56)
      ring = 2;
    else
      throw std::runtime_error("Ring not found in DynamicVetoProducer::getTrackerRegion2.");
    if (ring == 2)
      region += 3;
  } else
    throw std::runtime_error("SubdetectorId not found in DynamicVetoProducer::getTrackerRegion2.");

  return static_cast<TrackerRegion>(region);
}

// TrackerRegion DynamicVetoProducer::getTrackerRegion3(unsigned int mId){

//   auto detId = DetId(mId);
//   int subdetectorId = detId.subdetId();

//   int region = 0;
//   if ( subdetectorId == 1 ){
//     region = PixelBarrelName(detId, trackerTopology, false).layerName();
//   }
//   else if ( subdetectorId == 2 ){
//     auto tmp = PixelEndcapName(detId, trackerTopology, false);
//     region = tmp.diskName() + 3*tmp.ringName() - 3;
//                                // ring number , according to radius, 1-lower, 2- higher
//   }
//   return static_cast<TrackerRegion>(region);
// }

//--------------------------------------------------------------------------------------------------
int DynamicVetoProducer::addBadModules(const std::map<int, double>& moduleID2value,
                                       const double center,
                                       const double distance,
                                       std::vector<int>& badModules) {
  int countBad = 0;
  double thr1 = (center - distance);
  double thr2 = (center + distance);
  for (const auto& [key, value] : moduleID2value) {
    if ((value < thr1) || (thr2 < value)) {
      badModules.push_back(key);
      countBad++;
    }
  }
  return countBad;
}

//--------------------------------------------------------------------------------------------------

void DynamicVetoProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  lumisectionCount_++;
  if (coutOn_)
    std::cout << "DynamicVetoProducer::beginLuminosityBlock " << lumiSeg.luminosityBlock() << std::endl;
}

void DynamicVetoProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  if (coutOn_)
    std::cout << "DynamicVetoProducer::endLuminosityBlock " << lumiSeg.luminosityBlock() << std::endl;

  const edm::Handle<reco::PixelClusterCounts> pccHandle = lumiSeg.getHandle(pccToken_);
  const reco::PixelClusterCounts& inputPcc = *(pccHandle.product());

  //vector with Module IDs 1-1 map to bunch x-ing in clusers
  auto modID = inputPcc.readModID();
  //cluster counts per module per bx
  auto clustersPerBXInput = inputPcc.readCounts();

  std::map<TrackerRegion, double>
      region2meanClusterCount;  // orbit integrated average (over modules) cluster count in this LS
  std::map<TrackerRegion, unsigned int> region2moduleCount;
  for (size_t i = 0; i < modID.size(); i++) {
    if (std::find(baseVeto_.begin(), baseVeto_.end(), modID.at(i)) != baseVeto_.end())
      continue;
    // TrackerRegion region = TrackerRegion::Any;
    TrackerRegion region = getTrackerRegion1(modID.at(i));

    for (size_t bx = 0; bx < LumiConstants::numBX; bx++)
      region2meanClusterCount[region] += clustersPerBXInput.at(i * LumiConstants::numBX + bx);
    // for (size_t bx : filledBunches) region2meanClusterCount[region] += clustersPerBXInput.at( i * LumiConstants::numBX + bx );
    region2moduleCount[region]++;
  }
  for (const auto& [region, value] : region2meanClusterCount)
    region2meanClusterCount[region] /= region2moduleCount[region];

  for (size_t i = 0; i < modID.size(); i++) {
    if (std::find(baseVeto_.begin(), baseVeto_.end(), modID.at(i)) != baseVeto_.end())
      continue;
    // TrackerRegion region = TrackerRegion::Any;
    TrackerRegion region = getTrackerRegion1(modID.at(i));

    double bxsum = 0;
    for (int bx = 0; bx < int(LumiConstants::numBX); bx++)
      bxsum += clustersPerBXInput.at(i * LumiConstants::numBX + bx);
    // for (size_t bx : filledBunches) bxsum += clustersPerBXInput.at( i * LumiConstants::numBX + bx);
    region2moduleID2countRatio[region][modID.at(i)] += bxsum / region2meanClusterCount[region];
    region2moduleID2LS2counts[region][modID.at(i)].push_back(bxsum);
  }
}

//--------------------------------------------------------------------------------------------------
// void DynamicVetoProducer::dqmEndRun(const edm::Run & runSeg, const edm::EventSetup& iSetup) {
void DynamicVetoProducer::dqmEndRun(edm::Run& runSeg, const edm::EventSetup& iSetup) {
  if ((lumisectionCountMin_ > 0) && (lumisectionCount_ < lumisectionCountMin_)) {
    edm::LogInfo("INFO") << "Number of Lumisections " << lumisectionCount_ << " in run " << runSeg.run()
                         << " which is too few. Skipping update to veto list.";
    if (coutOn_)
      std::cout << "Number of Lumisections " << lumisectionCount_ << " in run " << runSeg.run()
                << " which is too few. Skipping update to veto list." << std::endl;
    resetContainers();
    return;
  }

  edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Number of Lumisections processed in run " << runSeg.run()
                       << " : " << lumisectionCount_;
  if (coutOn_)
    std::cout << "DynamicVetoProducer::dqmEndRun: Number of Lumisections processed in run " << runSeg.run() << " : "
              << lumisectionCount_ << std::endl;

  // round1: remove outliers in terms of occupancy in a given layer/region
  if (filterLevel_ >= 1)
    for (const auto& [region, moduleID2value] : region2moduleID2countRatio) {
      // center    = getMean(moduleID2value);
      // std       = getStd(moduleID2value, mean) ;

      auto quantiles = getQuantile(moduleID2value, {0.16, 0.5, 0.84});
      double center = quantiles[1];
      double std = std::min(quantiles[2] - quantiles[1], quantiles[1] - quantiles[0]);

      double distance = std * stdMultiplyer1_;
      int badModuleCount = addBadModules(moduleID2value, center, distance, additionalVeto1_);

      if (savePlots_) {
        std::string name = "Round1_REG" + std::to_string(int(region)) + "_RUN" + std::to_string(runSeg.run());
        makePlot(name, moduleID2value, center, std, distance, badModuleCount);
      }
    }
  // edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto1_.size();
  if (coutOn_)
    std::cout << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto1_.size() << std::endl;

  // round2: filter based on the stability of the per-LS cluster count of the module over the run
  if (filterLevel_ >= 2)
    for (const auto& [region, moduleID2LS2counts] : region2moduleID2LS2counts) {
      // recomputed average (over modules) number of clusters over not excluded moules in layer
      std::vector<double> LS2meanCounts = std::vector<double>(lumisectionCount_, 0);
      unsigned int moduleCount = 0;
      for (const auto& [mId, LS2counts] : moduleID2LS2counts) {
        if (std::find(additionalVeto1_.begin(), additionalVeto1_.end(), mId) != additionalVeto1_.end())
          continue;
        for (size_t i = 0; i < LS2counts.size(); i++)
          LS2meanCounts[i] += LS2counts[i];
        moduleCount++;
      }
      for (size_t i = 0; i < LS2meanCounts.size(); i++)
        LS2meanCounts[i] /= moduleCount;

      // find if too many are outliers
      for (const auto& [mId, LS2counts] : moduleID2LS2counts) {
        if (std::find(additionalVeto1_.begin(), additionalVeto1_.end(), mId) != additionalVeto1_.end())
          continue;

        // we must normalize to avoid spread due to burnoff during the run
        std::vector<double> data;
        for (size_t i = 0; i < LS2counts.size(); i++) {
          data.push_back(LS2counts[i]);
          data[i] /= LS2meanCounts[i];
        }

        auto quantiles = getQuantile(data, {0.16, 0.5, 0.84});
        double center = quantiles[1];
        double std = std::min(quantiles[2] - quantiles[1], quantiles[1] - quantiles[0]);

        double distance = std * stdMultiplyer2_;

        int countBad = 0;
        double thr1 = (center - distance);
        double thr2 = (center + distance);
        for (auto value : data) {
          if ((value < thr1) || (thr2 < value))
            countBad++;
        }

        double fractionBad = double(countBad) / LS2counts.size();
        if (fractionBad > fractionThreshold2_)
          additionalVeto2_.push_back(mId);  // should only be 0.2% for a gaussian distribution
      }
    }
  // edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Modules removed in round 2: " << additionalVeto2_.size();
  if (coutOn_)
    std::cout << "DynamicVetoProducer::dqmEndRun: Modules removed in round 2: " << additionalVeto2_.size() << std::endl;

  // round3: filter based on the fractional response of the module
  if (filterLevel_ >= 3) {
    double totalFraction = 0;
    double totalNumberOfClusters = 0;
    std::map<int, double> moduleID2totalNumberOfClusters;
    for (const auto& [region, moduleID2LS2counts] : region2moduleID2LS2counts) {
      for (const auto& [mId, LS2counts] : moduleID2LS2counts) {
        if (std::find(additionalVeto1_.begin(), additionalVeto1_.end(), mId) != additionalVeto1_.end())
          continue;
        if (std::find(additionalVeto2_.begin(), additionalVeto2_.end(), mId) != additionalVeto2_.end())
          continue;

        auto it = fractionalResponseMap.find(mId);
        if (it == fractionalResponseMap.end())
          throw std::runtime_error("Module not found in fractionalResponseMap");
        totalFraction += it->second;
        for (auto c : LS2counts) {
          totalNumberOfClusters += c;
          moduleID2totalNumberOfClusters[mId] += c;
        }
      }
    }

    std::map<int, double> moduleID2doubleRatio;
    for (const auto& [mId, moduleTotalCounts] : moduleID2totalNumberOfClusters) {
      moduleID2doubleRatio[mId] =
          (moduleTotalCounts / totalNumberOfClusters) / (fractionalResponseMap.at(mId) / totalFraction);
    }

    auto quantiles = getQuantile(moduleID2doubleRatio, {0.16, 0.5, 0.84});
    double center = quantiles[1];
    double std = std::min(quantiles[2] - quantiles[1], quantiles[1] - quantiles[0]);

    double distance = std * stdMultiplyer3_;
    int badModuleCount = addBadModules(moduleID2doubleRatio, center, distance, additionalVeto3_);

    if (savePlots_) {
      std::string name = "Round3_RUN" + std::to_string(runSeg.run());
      makePlot(name, moduleID2doubleRatio, center, std, distance, badModuleCount);
    }

    if (coutOn_)
      std::cout << "DynamicVetoProducer::dqmEndRun: Modules removed in round 3: " << additionalVeto3_.size()
                << std::endl;
  }

  // for a pretty output
  std::sort(additionalVeto1_.begin(), additionalVeto1_.end());
  std::sort(additionalVeto2_.begin(), additionalVeto2_.end());
  std::sort(additionalVeto3_.begin(), additionalVeto3_.end());

  // outoputs
  if (saveCSVFile_) {
    std::lock_guard<std::mutex> lock(fileLock_);
    std::ofstream csfile(csvOutLabel_, std::ios_base::app);

    csfile << std::to_string(runSeg.run()) << ",r1";
    for (auto v : additionalVeto1_)
      csfile << "," << std::to_string(v);
    csfile << std::endl;

    csfile << std::to_string(runSeg.run()) << ",r2";
    for (auto v : additionalVeto2_)
      csfile << "," << std::to_string(v);
    csfile << std::endl;

    csfile << std::to_string(runSeg.run()) << ",r3";
    for (auto v : additionalVeto3_)
      csfile << "," << std::to_string(v);
    csfile << std::endl;

    // for (const auto& [region, badModuleCount] : region2badModuleCount) {
    //   csfile << std::to_string(region) <<","<< std::to_string(badModuleCount) <<","<< std::to_string(region2center[region]) <<","<< std::to_string(region2std[region]) << std::endl;
    // }

    csfile.close();
    edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: CSV created: " << csvOutLabel_;
    if (coutOn_)
      std::cout << "DynamicVetoProducer::dqmEndRun: CSV created: " << csvOutLabel_ << std::endl;
  }

  PccVetoList pccVetoList;
  if (saveBaseVeto_)
    pccVetoList.addToVetoList(baseVeto_);
  pccVetoList.addToVetoList(additionalVeto1_);
  pccVetoList.addToVetoList(additionalVeto2_);
  pccVetoList.addToVetoList(additionalVeto3_);
  if (saveBaseVeto_)
    pccVetoList.generateResponseFraction(fractionalResponseMap);
  else
    pccVetoList.generateResponseFraction(fractionalResponseMap, baseVeto_);
  runSeg.emplace(putToken_, std::move(static_cast<PccVetoListTransient>(pccVetoList)));

  if (poolDbService.isAvailable()) {
    // timetype=cms.untracked.string("runnumber"),  should be set up in the config
    cond::Time_t iovStart = (cond::Time_t)(runSeg.run());

    // Hash writeOneIOV(const T& payload, Time_t time, const std::string& recordName)
    poolDbService->writeOneIOV(pccVetoList, iovStart, "PccVetoListRcd");
    if (coutOn_)
      std::cout << "DynamicVetoProducer::dqmEndRun: written to DB " << std::endl;

  } else {
    edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: PoolDBService required.";
    if (coutOn_)
      std::cout << "DynamicVetoProducer::dqmEndRun: PoolDBService required." << std::endl;

    // throw std::runtime_error("PoolDBService required.");
  }

  // reset containers
  resetContainers();
}

void DynamicVetoProducer::makePlot(std::string name,
                                   std::map<int, double> moduleID2value,
                                   double center,
                                   double std,
                                   double distance,
                                   int badModuleCount) {
  // for this to work BuildFile.xml must have
  // <library file="*.cc" name="CalibrationLumiAlCaRecoProducersPlugins">
  //   <flags EDM_PLUGIN="1"/>
  //     <use name="rootcore"/>
  //     <use name="rootgraphics"/>
  //     <use name="rootgpad"/>
  //     <use name="roothist"/>
  // </library>

  if (coutOn_)
    std::cout << name << ", center:" << center << ", qstd: " << std << ", distance: " << distance << " N excluded "
              << badModuleCount << std::endl;

  TCanvas* canvas = new TCanvas("c", "c");

  double min = 1000000, max = 0;
  for (const auto& [key, value] : moduleID2value) {
    if (value > max)
      max = value;
    if (value < min)
      min = value;
  }
  min = std::min(min, center - distance);
  max = std::max(max, center + distance);
  double margin = (max - min) * 0.05;
  min -= margin;
  max += margin;

  TH1D* h = new TH1D(name.c_str(), name.c_str(), 60, min, max);
  for (const auto& [key, value] : moduleID2value)
    h->Fill(value);
  if (coutOn_) {
    for (const auto& [key, value] : moduleID2value)
      std::cout << value << ", ";
    std::cout << std::endl;
  }

  h->Draw();
  double maxVal = h->GetMaximum();

  TLine* cutUp = new TLine(center + distance, 0, center + distance, maxVal * 1.05);
  TLine* q_50 = new TLine(center, 0, center, maxVal * 1.05);
  TLine* cutDown = new TLine(center - distance, 0, center - distance, maxVal * 1.05);

  cutUp->SetLineColor(kRed);
  cutUp->SetLineWidth(2);

  q_50->SetLineColor(kGreen);
  q_50->SetLineWidth(2);

  cutDown->SetLineColor(kRed);
  cutDown->SetLineWidth(2);

  cutUp->Draw("same");
  q_50->Draw("same");
  cutDown->Draw("same");
  h->GetXaxis()->SetTitle("Val");
  h->GetYaxis()->SetTitle("Number of ROCs");
  // canvas->Print(name.c_str());
  canvas->SaveAs((name + ".png").c_str());
  delete canvas;
}

void DynamicVetoProducer::resetContainers() {
  edm::LogInfo("INFO") << "DynamicVetoProducer::resetContainers: Executing.";
  if (coutOn_)
    std::cout << "DynamicVetoProducer::resetContainers: Executing." << std::endl;

  additionalVeto1_.clear();
  region2moduleID2countRatio.clear();
  lumisectionCount_ = 0;

  additionalVeto2_.clear();
  region2moduleID2LS2counts.clear();

  additionalVeto3_.clear();
}

void DynamicVetoProducer::endJob() {
  edm::LogInfo("INFO") << "DynamicVetoProducer::endJob: Executing.";
  if (coutOn_)
    std::cout << "DynamicVetoProducer::endJob: Executing." << std::endl;
  // histoFile->Close();
}

DEFINE_FWK_MODULE(DynamicVetoProducer);
