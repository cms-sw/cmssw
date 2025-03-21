/**_________________________________________________________________

Author: Peter Major

Description:
Output: dynamic veto list per run
Input: per LS per module cluster count data

procedure:
The filtering of the bad modules is carried out in 2 rounds:
- round0: a base veto list is taken into account
The detector is segmented into layes which are handled differently. The endcap is further segmented into 2 rings in eta. 
- round1: 

new files for this feature:
"CondFormats/Luminosity/interface/PccVetoList.h"
'CondFormats/Luminosity/src/headers.h'
"CondFormats/DataRecord/interface/PccVetoListRcd.h"
"CondFormats/DataRecord/src/PccVetoListRcd.cc"

modified: 
CondFormats/Luminosity/src/classes_def.xml
CondFormats/Luminosity/test/BuildFile.xml
________________________________________________________________**/
#include <memory>
#include <string>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <map>
#include <utility>
#include <mutex>
#include <cmath>

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Luminosity/interface/PccVetoList.h"
#include "CondFormats/DataRecord/interface/PccVetoListRcd.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
// #include "TMath.h"
// #include "TH1.h"
// #include "TGraph.h"
// #include "TGraphErrors.h"
// #include "TFile.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

enum class TrackerRegion {
  Any,
  BPix_1, BPix_2, BPix_3, nBPix_4,
  Epix_1_ring1, Epix_2_ring1, Epix_3_ring1,
  Epix_1_ring2, Epix_2_ring2, Epix_3_ring2,
};

class DynamicVetoProducer : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit DynamicVetoProducer(const edm::ParameterSet&);
  ~DynamicVetoProducer() override;

private:

  // values to set up in the config
  edm::EDGetTokenT<reco::PixelClusterCounts> pccToken_;

  const std::map<int, std::string> moduleToRegionMap_;
  const std::vector<int> baseVeto_;
  const std::vector<int> moduleListRing1_;
  const int lumisectionCountMin_;
  const bool stdMultiplyer1_;
  const bool stdMultiplyer2_;

  // working containers
  //// for round 1
  std::vector<int> additionalVeto1_;
  std::map<TrackerRegion, std::map<int, double> > region2moduleID2countRatio;
  unsigned int lumisectionCount_ = 0;
  // std::map<TrackerRegion, double > region2center;
  // std::map<TrackerRegion, double > region2std;
  // std::map<TrackerRegion, int >   region2badModuleCount;

  //// for round 2
  std::vector<int> additionalVeto2_;
  std::map<TrackerRegion, std::map<int, std::vector<unsigned int> > > region2moduleID2LS2counts;


  // TFile* histoFile;
  // std::map<TrackerRegion, TH1F* > region2countHistogram;

  // geometry
  TrackerRegion getTrackerRegion(unsigned int mId);

  //produce csv lumi file
  const bool saveCSVFile_;
  const std::string csvOutLabel_;
  mutable std::mutex fileLock_;

  // math
  double getMean(const std::map<int, unsigned int>& moduleID2value);
  double getStd(const std::map<int, unsigned int>& moduleID2value, const double mean);
  double interp(double v0, double v1, double t) { return (1 - t)*v0 + t*v1; }
  std::vector<double> getQuantile(const std::vector<double>& inData, const std::vector<double>& probs);
  std::vector<double> getQuantile(const std::map<int, double >& inData, const std::vector<double>& probs);

  // select bad modules
  int addBadModules(const std::map<int, double>& moduleID2value, const double mean, const double distance,  std::vector<int>& badModules);

  // actions
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void dqmEndRun(edm::Run const& runSeg, const edm::EventSetup& iSetup) final;
  void dqmEndRunProduce(const edm::Run& runSeg, const edm::EventSetup& iSetup);
  void endJob() final;

  // must have
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& context) {}

  // DB Service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
};

//--------------------------------------------------------------------------------------------------
DynamicVetoProducer::DynamicVetoProducer(const edm::ParameterSet& iConfig)
    : pccToken_(consumes<reco::PixelClusterCounts, edm::InLumi>(edm::InputTag(
          iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getParameter<std::string>("inputPccLabel")))),
      // moduleToRegionMap_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getParameter< std::map<int, std::string> >("ModuleToRegionMap")),
      baseVeto_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getParameter<std::vector<int> >("BaseVeto")),
      moduleListRing1_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getUntrackedParameter<std::vector<int> >("ModuleListRing1", {344724484, 344725508, 344728580, 344729604, 344732676, 344733700, 344736772, 344737796, 344740868, 344741892, 344744964, 344745988, 344749060, 344750084, 344753156, 344754180, 344757252, 344758276, 344761348, 344762372, 344765444, 344766468, 344769540, 344770564, 344773636, 344774660, 344777732, 344778756, 344781828, 344782852, 344785924, 344786948, 344790020, 344791044, 344794116, 344795140, 344798212, 344799236, 344802308, 344803332, 344806404, 344807428, 344810500, 344811524, 344462340, 344463364, 344466436, 344467460, 344470532, 344471556, 344474628, 344475652, 344478724, 344479748, 344482820, 344483844, 344486916, 344487940, 344491012, 344492036, 344495108, 344496132, 344499204, 344500228, 344503300, 344504324, 344507396, 344508420, 344511492, 344512516, 344515588, 344516612, 344519684, 344520708, 344523780, 344524804, 344527876, 344528900, 344531972, 344532996, 344536068, 344537092, 344540164, 344541188, 344544260, 344545284, 344548356, 344549380, 344200196, 344201220, 344204292, 344949764, 344208388, 344818692, 344212484, 344830980, 344216580, 344217604, 344220676, 344843268, 344224772, 344225796, 344228868, 344851460, 344232964, 344859652, 344237060, 344238084, 344241156, 344871940, 344245252, 344246276, 344249348, 344880132, 344253444, 344888324, 344257540, 344900612, 344261636, 344262660, 344265732, 344912900, 344269828, 344270852, 344273924, 344921092, 344278020, 344929284, 344282116, 344283140, 344286212, 344941572, 352588804, 352589828, 352592900, 353215492, 352596996, 352598020, 352601092, 353227780, 352605188, 353235972, 352609284, 352610308, 352613380, 353244164, 352617476, 352618500, 352621572, 353256452, 352625668, 353268740, 352629764, 353276932, 352633860, 352634884, 352637956, 353285124, 352642052, 352643076, 352646148, 353297412, 352650244, 353305604, 352654340, 352655364, 352658436, 353313796, 352662532, 352663556, 352666628, 353326084, 352670724, 353338372, 352674820, 353207300, 352850948, 352851972, 352855044, 352856068, 352859140, 352860164, 352863236, 352864260, 352867332, 352868356, 352871428, 352872452, 352875524, 352876548, 352879620, 352880644, 352883716, 352884740, 352887812, 352888836, 352891908, 352892932, 352896004, 352897028, 352900100, 352901124, 352904196, 352905220, 352908292, 352909316, 352912388, 352913412, 352916484, 352917508, 352920580, 352921604, 352924676, 352925700, 352928772, 352929796, 352932868, 352933892, 352936964, 352937988, 353113092, 353114116, 353117188, 353118212, 353121284, 353122308, 353125380, 353126404, 353129476, 353130500, 353133572, 353134596, 353137668, 353138692, 353141764, 353142788, 353145860, 353146884, 353149956, 353150980, 353154052, 353155076, 353158148, 353159172, 353162244, 353163268, 353166340, 353167364, 353170436, 353171460, 353174532, 353175556, 353178628, 353179652, 353182724, 353183748, 353186820, 353187844, 353190916, 353191940, 353195012, 353196036, 353199108, 353200132})),
      lumisectionCountMin_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getUntrackedParameter<int>("MinimumLSCount", 10)),
      stdMultiplyer1_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getParameter<double>("StdMultiplyier1")),
      stdMultiplyer2_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getParameter<double>("StdMultiplyier2")),
      saveCSVFile_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getUntrackedParameter<bool>("SaveCSVFile", false)),
      csvOutLabel_(iConfig.getParameter<edm::ParameterSet>("DynamicVetoProducerParameters").getUntrackedParameter<std::string>("CsvFileName", std::string("dynamicVeto.csv"))) 
      {

      }

//--------------------------------------------------------------------------------------------------
DynamicVetoProducer::~DynamicVetoProducer() {}

//--------------------------------------------------------------------------------------------------
double DynamicVetoProducer::getMean(const std::map<int, unsigned int>& moduleID2value){
  double sum = 0;
  for (const auto& [key, value] : moduleID2value) {
    sum += value;
  }
  return sum/moduleID2value.size();
}

double DynamicVetoProducer::getStd(const std::map<int, unsigned int>& moduleID2value, const double mean){
  double sum2 = 0;
  for (const auto& [key, value] : moduleID2value) {
    sum2 += std::pow(value, 2);
  }
  return std::pow(sum2/moduleID2value.size()-mean*mean, 0.5);
}

std::vector<double> DynamicVetoProducer::getQuantile(const std::vector<double>& inData, const std::vector<double>& probs)
{
  if (inData.empty() || probs.empty()) return std::vector<double>();
  if (1 == inData.size()) return std::vector<double>(probs.size(), inData[0]);

  std::vector<double> data = inData;
  std::sort(data.begin(), data.end());

  std::vector<double> quantiles;
  for (size_t i = 0; i < probs.size(); ++i)
  {
      double target = interp(-0.5, data.size() - 0.5, probs[i]);

      size_t left = std::max(int64_t(std::floor(target)), int64_t(0));
      size_t right = std::min(int64_t(std::ceil(target)), int64_t(data.size() - 1));

      double datLeft = data.at(left);
      double datRight = data.at(right);

      quantiles.push_back( interp(datLeft, datRight, target - left) );
  }

  return quantiles;
}

std::vector<double> DynamicVetoProducer::getQuantile(const std::map<int, double >& inData, const std::vector<double>& probs)
{
  if (inData.empty() || probs.empty()) return std::vector<double>();
  if (1 == inData.size()) return std::vector<double>(probs.size(), inData.begin()->second );

  std::vector<double> data;
  for (const auto& [key, value] : inData) data.push_back(value);
  std::sort(data.begin(), data.end());

  std::vector<double> quantiles;
  for (size_t i = 0; i < probs.size(); ++i)
  {
      double target = interp(-0.5, data.size() - 0.5, probs[i]);

      size_t left = std::max(int64_t(std::floor(target)), int64_t(0));
      size_t right = std::min(int64_t(std::ceil(target)), int64_t(data.size() - 1));

      double datLeft = data.at(left);
      double datRight = data.at(right);

      quantiles.push_back( interp(datLeft, datRight, target - left) );
  }

  return quantiles;
}

TrackerRegion DynamicVetoProducer::getTrackerRegion(unsigned int mId){
  //https://gitlab.cern.ch/cms-sw/cmssw/tree/7eb5fd3cd39b94ea5618c5c177817c10c7675428/Geometry/TrackerNumberingBuilder
  // I tested all BPix modules by bitwis ANDing ring 1 and ring 2 moduleIDs separately. The results are the same, therefore must use LUT.
  // ring 1 has fewer elements, so that is used
  int layer = 0;
  unsigned int subdetectorId = ((mId>>25) & 0x7);
  // int subid = DetId(mod).subdetId();
  if ( subdetectorId == 1 ){
    layer = ((mId>>20) & 7 );
  }
  else if ( subdetectorId == 2 ){
    layer = ((mId>>18) & 7 );
    layer += 4;
    if (std::find(moduleListRing1_.begin(), moduleListRing1_.end(), mId) == moduleListRing1_.end()) layer+=3;
  }
  return static_cast<TrackerRegion>(layer);
}

//--------------------------------------------------------------------------------------------------
int DynamicVetoProducer::addBadModules(const std::map<int, double>& moduleID2value, const double center, const double distance,  std::vector<int>& badModules){
  int countBad = 0;
  double thr1 = (center - distance);
  double thr2 = (center + distance);
  for (const auto& [key, value] : moduleID2value) {
    if ((value < thr1) || (thr2 < value) ) {
      badModules.push_back(key);
      countBad ++;
    }
  }
  return countBad;
}

//--------------------------------------------------------------------------------------------------

void DynamicVetoProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  lumisectionCount_++;
}

void DynamicVetoProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {

  const edm::Handle<reco::PixelClusterCounts> pccHandle = lumiSeg.getHandle(pccToken_);
  const reco::PixelClusterCounts& inputPcc = *(pccHandle.product());
  
  //vector with Module IDs 1-1 map to bunch x-ing in clusers
  auto modID = inputPcc.readModID();
  //cluster counts per module per bx
  auto clustersPerBXInput = inputPcc.readCounts();



  std::map<TrackerRegion, double > region2meanClusterCount; // orbit integrated average cluster count in this LS
  std::map<TrackerRegion, unsigned int > region2moduleCount;
  for (size_t i = 0; i < modID.size(); i++) {
    // std::cout<<","<<modID.at(i)<<std::endl;
    if (std::find(baseVeto_.begin(), baseVeto_.end(), modID.at(i)) != baseVeto_.end()) continue;
    // TrackerRegion region = TrackerRegion::Any;
    TrackerRegion region = getTrackerRegion( modID.at(i) );
    

    for (size_t bx = 0; bx < LumiConstants::numBX; bx++) {
      region2meanClusterCount[region] += clustersPerBXInput.at( i * LumiConstants::numBX + bx );
    }
    region2moduleCount[region] ++;
  }
  for (const auto& [region, value] : region2meanClusterCount) region2meanClusterCount[region] /= region2moduleCount[region];



  for (size_t i = 0; i < modID.size(); i++) {
    if (std::find(baseVeto_.begin(), baseVeto_.end(), modID.at(i)) != baseVeto_.end()) continue;
    // TrackerRegion region = TrackerRegion::Any;
    TrackerRegion region = getTrackerRegion( modID.at(i) );

    double bxsum = 0;
    for (int bx = 0; bx < int(LumiConstants::numBX); bx++) bxsum += clustersPerBXInput.at( i * int(LumiConstants::numBX) + bx);
    region2moduleID2countRatio[region][modID.at(i)] += bxsum / region2moduleCount[region];
    region2moduleID2LS2counts[region][modID.at(i)].push_back( bxsum );
  }

  // debug
  for (const auto& [key, value] : region2moduleID2countRatio){
    int k = static_cast<int>(key);
    std::cout << k << " " << value.size() << std::endl;
    if (true){
      for (const auto& [key2, value2] : value){
        std::cout << ", " << key2;
      }
    }
    std::cout<<std::endl;
  } 

}

//--------------------------------------------------------------------------------------------------
void DynamicVetoProducer::dqmEndRun(edm::Run const& runSeg, const edm::EventSetup& iSetup) {

  // if ( lumisectionCount_ < lumisectionCountMin_ ) {
  //   edm::LogInfo("INFO") << "Number of Lumisections " << lumisectionCount_ << " in run " << runSeg.run() << " which is too few. Skipping update to veto list.";
  //   return;
  // }

  edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Number of Lumisections processed in run " << runSeg.run() << " : " << lumisectionCount_;
  std::cout << "DynamicVetoProducer::dqmEndRun: Number of Lumisections processed in run " << runSeg.run() << " : " << lumisectionCount_ << std::endl;


  // round1: remove outliers in terms of occupancy
  for (const auto& [region, moduleID2value] : region2moduleID2countRatio) {
    
    // center    = getMean(moduleID2value);
    // std       = getStd(moduleID2value, mean) ;

    auto  quantiles = getQuantile(moduleID2value, {0.16, 0.5, 0.84});
    double center    = quantiles[1];
    double std       = std::min(quantiles[2]-quantiles[1], quantiles[1]-quantiles[0]);

    double distance  = std * stdMultiplyer1_;
    int badModuleCount = addBadModules(moduleID2value, center, distance, additionalVeto1_);

    // region2center[region] = center;
    // region2std[region]  = std;
    // region2badModuleCount[region] = badModuleCount;
  }
  // edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto1_.size();
  std::cout << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto1_.size() << std::endl;



  // round2:  
  for (const auto& [region, moduleID2LS2counts] : region2moduleID2LS2counts) {

    // average number of clusters over not excluded moules in layer
    std::vector<double> LS2meanCounts = std::vector<double>(lumisectionCount_, 0);
    unsigned int moduleCount = 0;
    for (const auto& [mId, LS2counts] : moduleID2LS2counts) {
      if (std::find(additionalVeto1_.begin(), additionalVeto1_.end(), mId) != additionalVeto1_.end()) continue;
      for (size_t i=0; i<LS2counts.size(); i++) LS2meanCounts[i] += LS2counts[i];
      moduleCount ++;
    }
    for (size_t i=0; i<LS2meanCounts.size(); i++) LS2meanCounts[i] /=moduleCount;

    // find if too many are outliers
    for (const auto& [mId, LS2counts] : moduleID2LS2counts) {
      if (std::find(additionalVeto1_.begin(), additionalVeto1_.end(), mId) != additionalVeto1_.end()) continue;

      // we must normalize to avoid spread due to burnoff during the run
      std::vector<double> data;
      for (size_t i=0; i<LS2counts.size(); i++){
        data.push_back(LS2counts[i]);
        data[i] /= LS2meanCounts[i];
      }

      auto  quantiles = getQuantile(data, {0.16, 0.5, 0.84});
      double center    = quantiles[1];
      double std       = std::min(quantiles[2]-quantiles[1], quantiles[1]-quantiles[0]);

      double distance  = std * stdMultiplyer2_;

      int countBad = 0;
      double thr1 = (center - distance);
      double thr2 = (center + distance);
      for (auto value : data) {
        if ((value < thr1) || (thr2 < value) ) countBad ++;
      }

      double fractionBad = double(countBad) / LS2counts.size();
      if(fractionBad>0.02) additionalVeto2_.push_back( mId ); // should only be 0.2% for a gaussian distribution
    }
  }
  // edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto2_.size();
  std::cout << "DynamicVetoProducer::dqmEndRun: Modules removed in round 1: " << additionalVeto2_.size() << std::endl;



  // for a pretty output
  std::sort(additionalVeto1_.begin(), additionalVeto1_.end());
  std::sort(additionalVeto2_.begin(), additionalVeto2_.end());

  // outoputs
  if (saveCSVFile_) {
    std::lock_guard<std::mutex> lock(fileLock_);
    std::ofstream csfile(csvOutLabel_, std::ios_base::app);

    csfile << std::to_string(runSeg.run())<<",r1";
    for (auto v : additionalVeto1_) csfile <<","<< std::to_string(v);
    csfile << std::endl;

    csfile << std::to_string(runSeg.run())<<",r2";
    for (auto v : additionalVeto2_) csfile <<","<< std::to_string(v);
    csfile << std::endl;

    // for (const auto& [region, badModuleCount] : region2badModuleCount) {
    //   csfile << std::to_string(region) <<","<< std::to_string(badModuleCount) <<","<< std::to_string(region2center[region]) <<","<< std::to_string(region2std[region]) << std::endl;
    // }

    csfile.close();
    // edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: CSV created.";
  }

  if (poolDbService.isAvailable()) {
    //Writing the corrections to SQL lite file for db
    PccVetoList pccVetoList;
    pccVetoList.addToVetoList(baseVeto_);
    pccVetoList.addToVetoList(additionalVeto1_);
    pccVetoList.addToVetoList(additionalVeto2_);

    // TODO: is this the right "time value" to use?
    cond::Time_t thisIOV = (cond::Time_t)(runSeg.run()); // IOV: interval of validity

    // Hash writeOneIOV(const T& payload, Time_t time, const std::string& recordName)
    poolDbService->writeOneIOV(pccVetoList, thisIOV, "PccVetoListRcd");
  } else {
    edm::LogInfo("INFO") << "DynamicVetoProducer::dqmEndRun: PoolDBService required.";
    // throw std::runtime_error("PoolDBService required.");
  }

  // reset containers
  additionalVeto1_.clear();
  region2moduleID2countRatio.clear();
  lumisectionCount_ = 0;

  additionalVeto2_.clear();
  region2moduleID2LS2counts.clear();
}


void DynamicVetoProducer::endJob() {
  edm::LogInfo("INFO") << "DynamicVetoProducer::endJob: Executing.";
}

DEFINE_FWK_MODULE(DynamicVetoProducer);
