#include "RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTester.h"

#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <sstream>

void 
ClusterizerUnitTester::
analyze(const edm::Event&, const edm::EventSetup& es) {
  for(iter_t group = testGroups.begin(); group < testGroups.end(); group++) {
    initializeTheGroup(*group,es);
    testTheGroup(*group);
  }
}

void
ClusterizerUnitTester::
initializeTheGroup(const PSet& group, const edm::EventSetup& es) {
  PSet params = group.getParameter<PSet>("ClusterizerParameters");
  if(clusterizer) 
    delete clusterizer;
  clusterizer = 
    new ThreeThresholdStripClusterizer( params.getParameter<double>("Channel"),
					params.getParameter<double>("Seed"),
					params.getParameter<double>("Cluster"),
					params.getParameter<unsigned>("MaxSequentialHoles"),
					params.getParameter<unsigned>("MaxSequentialBad"),
					params.getParameter<unsigned>("MaxAdjacentBad"));
  clusterizer->init(es);
}

void 
ClusterizerUnitTester::
testTheGroup(const PSet& group) {
  std::string label = group.getParameter<std::string>("Label");
  PSet params = group.getParameter<PSet>("ClusterizerParameters");
  VPSet tests = group.getParameter<VPSet>("Tests");

  std::cout << "\nTesting group: \"" << label << "\"\n               " << printParams(params) << std::endl;
  for(iter_t test = tests.begin();  test < tests.end();  test++) {
    runTheTest(*test);
    detId++;
  }
}

std::string
ClusterizerUnitTester::
printParams(const PSet& params) {
  std::stringstream s;
  s << "Channel(" << params.getParameter<double>("Channel") <<") "
    << "Seed("    << params.getParameter<double>("Seed")    <<") "
    << "Cluster(" << params.getParameter<double>("Cluster") <<") "
    << "MaxSequentialHoles("<< params.getParameter<unsigned>("MaxSequentialHoles") <<") "
    << "MaxSequentialBad("<< params.getParameter<unsigned>("MaxSequentialBad") <<") "
    << "MaxAdjacentBad("<< params.getParameter<unsigned>("MaxAdjacentBad") <<")\n";
  return s.str();
}

void 
ClusterizerUnitTester::
runTheTest(const PSet& test) {
  std::string label =  test.getParameter<std::string>("Label");
  VPSet clusterset = test.getParameter<VPSet>("Clusters");
  VPSet digiset    = test.getParameter<VPSet>("Digis");
  std::cout << "Testing: \"" << label << "\"\n";

  edm::DetSet<SiStripDigi> digis(detId); 
  edmNew::DetSetVector<SiStripCluster> expected;
  edmNew::DetSetVector<SiStripCluster> result;
  edmNew::DetSetVector<SiStripCluster>::FastFiller expectedFF(expected, detId);
  edmNew::DetSetVector<SiStripCluster>::FastFiller resultFF(result, detId);

  constructDigis(digiset, digis);
  constructClusters(clusterset, expectedFF);  
  if(expectedFF.empty()) expectedFF.abort();
  
  try { 
    clusterizer->clusterizeDetUnit(digis, resultFF); 
    if(resultFF.empty()) resultFF.abort();
    assertIdentical(expected, result);
  }
  catch(ThreeThresholdStripClusterizer::InvalidChargeException e) {
    if(!test.getParameter<bool>("InvalidCharge")) throw e;
  }
  catch(cms::Exception e) {
    std::cout << ( e << "Input:\n" << printDigis(digiset));
  }
}

void
ClusterizerUnitTester::
constructDigis(const VPSet& stripset, edm::DetSet<SiStripDigi>& digis) {
  for(iter_t strip = stripset.begin(); strip < stripset.end(); strip++) {
    digis.data.push_back( SiStripDigi(strip->getParameter<unsigned>("Strip"),
				      strip->getParameter<unsigned>("ADC") ));
  }
}

std::string
ClusterizerUnitTester::
printDigis(const VPSet& stripset){
  std::stringstream s;
  for(iter_t strip = stripset.begin(); strip < stripset.end(); strip++) {
    s << "\t(" 
      <<  strip->getParameter<unsigned>("Strip") << ", "
      <<  strip->getParameter<unsigned>("ADC")   << ", "
      <<  strip->getParameter<double>("Noise")   << ", "
      <<  strip->getParameter<double>("Gain")    << ", "
      << ( strip->getParameter<bool>("Quality")  ? "good" : "bad")
      << " )\n";
  }
  return s.str();
}

void
ClusterizerUnitTester::
constructClusters(const VPSet& clusterset, 
		  edmNew::DetSetVector<SiStripCluster>::FastFiller& clusters) {
  for(iter_t c = clusterset.begin(); c<clusterset.end(); c++) {
    uint16_t firststrip =  c->getParameter<unsigned>("FirstStrip");
    std::vector<unsigned> amplitudes =  c->getParameter<std::vector<unsigned> >("Amplitudes");
    std::vector<uint16_t> a16(amplitudes.begin(),amplitudes.end());
    clusters.push_back(SiStripCluster(detId, firststrip, a16.begin(),a16.end()));
  }
}

void
ClusterizerUnitTester::
assertIdentical(const edmNew::DetSetVector<SiStripCluster>& L, 
		const edmNew::DetSetVector<SiStripCluster>& R) {
  if(!clusterDSVsIdentical(L,R))
    throw cms::Exception("Failed") << "Expected:\n" << printDSV(L) 
				   << "Actual:\n"   << printDSV(R);
}

bool
ClusterizerUnitTester::
clusterDSVsIdentical(const edmNew::DetSetVector<SiStripCluster>& L, 
		     const edmNew::DetSetVector<SiStripCluster>& R) {
  return 
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		       bool(true), std::logical_and<bool>(), clusterDetSetsIdentical );  
}

bool 
ClusterizerUnitTester::
clusterDetSetsIdentical(const edmNew::DetSet<SiStripCluster>& L, 
			const edmNew::DetSet<SiStripCluster>& R) {
  return 
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		       bool(true), std::logical_and<bool>(), clustersIdentical );
}

bool 
ClusterizerUnitTester::
clustersIdentical(const SiStripCluster& L, const SiStripCluster& R) {
  return
    L.geographicalId() == R.geographicalId()
    && L.firstStrip() == R.firstStrip() 
    && L.amplitudes().size() == R.amplitudes().size()
    && inner_product(L.amplitudes().begin(), L.amplitudes().end(), R.amplitudes().begin(), 
					 bool(true), std::logical_and<bool>(), std::equal_to<uint16_t>() );
}


std::string 
ClusterizerUnitTester::
printDSV(const edmNew::DetSetVector<SiStripCluster>& dsv) {
  std::stringstream s;
  for(edmNew::DetSetVector<SiStripCluster>::const_iterator 
	it = dsv.begin(); it<dsv.end(); it++)
    for(edmNew::DetSet<SiStripCluster>::const_iterator
	  cluster = it->begin(); cluster < it->end() ; cluster++)
      s << printCluster(*cluster);
  return s.str();
}

std::string 
ClusterizerUnitTester::
printCluster(const SiStripCluster & cluster) {
  std::stringstream s;
  s  << "\t" << cluster.firstStrip() << " [ ";
  for(unsigned i=0; i<cluster.amplitudes().size(); i++) {
    s << static_cast<int>(cluster.amplitudes().at(i)) << " ";
  }
  s << "]" << std::endl;
  return s.str();
}

