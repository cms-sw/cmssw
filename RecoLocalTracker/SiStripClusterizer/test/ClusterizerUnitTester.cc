// system includes
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>

// user includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class ClusterizerUnitTester : public edm::one::EDAnalyzer<> {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;
  typedef edmNew::DetSetVector<SiStripCluster> output_t;

public:
  explicit ClusterizerUnitTester(const PSet& conf);
  ~ClusterizerUnitTester() {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&);

  void initializeTheGroup(const PSet&, const edm::EventSetup&);
  void testTheGroup(const PSet&, const StripClusterizerAlgorithm*);
  void runTheTest(const PSet&, const StripClusterizerAlgorithm*);

  void constructClusters(const VPSet&, output_t&);
  void constructDigis(const VPSet&, edmNew::DetSetVector<SiStripDigi>&);

  static std::string printDigis(const VPSet&);
  static void assertIdentical(const output_t&, const output_t&);
  static bool clusterDSVsIdentical(const output_t&, const output_t&);
  static bool clusterDetSetsIdentical(const edmNew::DetSet<SiStripCluster>&, const edmNew::DetSet<SiStripCluster>&);
  static bool clustersIdentical(const SiStripCluster&, const SiStripCluster&);
  static std::string printDSV(const output_t&);
  static std::string printCluster(const SiStripCluster&);

  VPSet testGroups;
  std::vector<std::unique_ptr<StripClusterizerAlgorithm>> clusterizers;
  uint32_t detId;
};

ClusterizerUnitTester::ClusterizerUnitTester(const PSet& conf)
    : testGroups(conf.getParameter<VPSet>("ClusterizerTestGroups")) {
  for (const auto& group : testGroups) {
    clusterizers.push_back(StripClusterizerAlgorithmFactory::create(consumesCollector(),
                                                                    group.getParameter<PSet>("ClusterizerParameters")));
  }
}

void ClusterizerUnitTester::analyze(const edm::Event&, const edm::EventSetup& es) {
  detId = 0;
  for (std::size_t i = 0; i != testGroups.size(); ++i) {
    clusterizers[i]->initialize(es);
    testTheGroup(testGroups[i], clusterizers[i].get());
  }
}

void ClusterizerUnitTester::testTheGroup(const PSet& group, const StripClusterizerAlgorithm* clusterizer) {
  std::string label = group.getParameter<std::string>("Label");
  PSet params = group.getParameter<PSet>("ClusterizerParameters");
  VPSet tests = group.getParameter<VPSet>("Tests");

  std::cout << "\nTesting group: \"" << label << "\"\n               " << params << std::endl;
  for (const auto& test : tests) {
    runTheTest(test, clusterizer);
    detId++;
  }
}

void ClusterizerUnitTester::runTheTest(const PSet& test, const StripClusterizerAlgorithm* clusterizer) {
  std::string label = test.getParameter<std::string>("Label");
  VPSet clusterset = test.getParameter<VPSet>("Clusters");
  VPSet digiset = test.getParameter<VPSet>("Digis");
  std::cout << "Testing: \"" << label << "\"\n";

  edmNew::DetSetVector<SiStripDigi> digis;
  constructDigis(digiset, digis);

  output_t expected;
  constructClusters(clusterset, expected);

  output_t result;
  result.reserve(2 * clusterset.size(), 8 * clusterset.size());

  try {
    clusterizer->clusterize(digis, result);
    assertIdentical(expected, result);
    if (test.getParameter<bool>("InvalidCharge"))
      throw cms::Exception("Failed") << "Charges are valid, contrary to expectation.\n";
  } catch (StripClusterizerAlgorithm::InvalidChargeException const&) {
    if (!test.getParameter<bool>("InvalidCharge"))
      throw;
  } catch (cms::Exception& e) {
    std::cout << (e << "Input:\n" << printDigis(digiset));
  }
}

void ClusterizerUnitTester::constructDigis(const VPSet& stripset, edmNew::DetSetVector<SiStripDigi>& digis) {
  edmNew::DetSetVector<SiStripDigi>::TSFastFiller digisFF(digis, detId);
  for (iter_t strip = stripset.begin(); strip < stripset.end(); strip++) {
    digisFF.push_back(SiStripDigi(strip->getParameter<unsigned>("Strip"), strip->getParameter<unsigned>("ADC")));
  }
  if (digisFF.empty())
    digisFF.abort();
}

void ClusterizerUnitTester::constructClusters(const VPSet& clusterset, output_t& clusters) {
  output_t::TSFastFiller clustersFF(clusters, detId);
  for (iter_t c = clusterset.begin(); c < clusterset.end(); c++) {
    uint16_t firststrip = c->getParameter<unsigned>("FirstStrip");
    std::vector<unsigned> amplitudes = c->getParameter<std::vector<unsigned>>("Amplitudes");
    std::vector<uint16_t> a16(amplitudes.begin(), amplitudes.end());
    clustersFF.push_back(SiStripCluster(firststrip, a16.begin(), a16.end()));
  }
  if (clustersFF.empty())
    clustersFF.abort();
}

std::string ClusterizerUnitTester::printDigis(const VPSet& stripset) {
  std::stringstream s;
  for (iter_t strip = stripset.begin(); strip < stripset.end(); strip++) {
    s << "\t(" << strip->getParameter<unsigned>("Strip") << ", " << strip->getParameter<unsigned>("ADC") << ", "
      << strip->getParameter<double>("Noise") << ", " << strip->getParameter<double>("Gain") << ", "
      << (strip->getParameter<bool>("Quality") ? "good" : "bad") << " )\n";
  }
  return s.str();
}

void ClusterizerUnitTester::assertIdentical(const output_t& L, const output_t& R) {
  if (!clusterDSVsIdentical(L, R))
    throw cms::Exception("Failed") << "Expected:\n" << printDSV(L) << "Actual:\n" << printDSV(R);
}

bool ClusterizerUnitTester::clusterDSVsIdentical(const output_t& L, const output_t& R) {
  return L.size() == R.size() &&
         inner_product(L.begin(), L.end(), R.begin(), bool(true), std::logical_and<bool>(), clusterDetSetsIdentical);
}

bool ClusterizerUnitTester::clusterDetSetsIdentical(const edmNew::DetSet<SiStripCluster>& L,
                                                    const edmNew::DetSet<SiStripCluster>& R) {
  return L.size() == R.size() &&
         inner_product(L.begin(), L.end(), R.begin(), bool(true), std::logical_and<bool>(), clustersIdentical);
}

bool ClusterizerUnitTester::clustersIdentical(const SiStripCluster& L, const SiStripCluster& R) {
  return L.firstStrip() == R.firstStrip() && L.amplitudes().size() == R.amplitudes().size() &&
         inner_product(L.amplitudes().begin(),
                       L.amplitudes().end(),
                       R.amplitudes().begin(),
                       bool(true),
                       std::logical_and<bool>(),
                       std::equal_to<uint16_t>());
}

std::string ClusterizerUnitTester::printDSV(const output_t& dsv) {
  std::stringstream s;
  for (output_t::const_iterator it = dsv.begin(); it < dsv.end(); it++)
    for (edmNew::DetSet<SiStripCluster>::const_iterator cluster = it->begin(); cluster < it->end(); cluster++)
      s << printCluster(*cluster);
  return s.str();
}

std::string ClusterizerUnitTester::printCluster(const SiStripCluster& cluster) {
  std::stringstream s;
  s << "\t" << cluster.firstStrip() << " [ ";
  for (unsigned i = 0; i < cluster.amplitudes().size(); i++) {
    s << static_cast<int>(cluster.amplitudes()[i]) << " ";
  }
  s << "]" << std::endl;
  return s.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterizerUnitTester);
