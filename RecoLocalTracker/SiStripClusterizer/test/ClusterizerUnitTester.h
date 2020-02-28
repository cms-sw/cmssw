#ifndef ClusterizerUnitTester_h
#define ClusterizerUnitTester_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include <vector>
#include <string>
#include <memory>

class ClusterizerUnitTester : public edm::EDAnalyzer {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;
  typedef edmNew::DetSetVector<SiStripCluster> output_t;

public:
  ClusterizerUnitTester(const PSet& conf) : testGroups(conf.getParameter<VPSet>("ClusterizerTestGroups")) {}
  ~ClusterizerUnitTester() {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&);

  void initializeTheGroup(const PSet&, const edm::EventSetup&);
  void testTheGroup(const PSet&);
  void runTheTest(const PSet&);

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
  std::unique_ptr<StripClusterizerAlgorithm> clusterizer;
  uint32_t detId;
};

#endif
