#include "RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTester.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <sstream>

void ClusterizerUnitTester::
analyze(const edm::Event&, const edm::EventSetup& es) {
  detId=0;
  for(iter_t group = testGroups.begin(); group < testGroups.end(); group++) {
    clusterizer = StripClusterizerAlgorithmFactory::create(group->getParameter<PSet>("ClusterizerParameters"));
    clusterizer->initialize(es);
    testTheGroup(*group);
  }
}

void ClusterizerUnitTester::
testTheGroup(const PSet& group) {
  std::string label = group.getParameter<std::string>("Label");
  PSet params = group.getParameter<PSet>("ClusterizerParameters");
  VPSet tests = group.getParameter<VPSet>("Tests");

  std::cout << "\nTesting group: \"" << label << "\"\n               " << params << std::endl;
  for(iter_t test = tests.begin();  test < tests.end();  test++) {
    runTheTest(*test);
    detId++;
  }
}

void ClusterizerUnitTester::
runTheTest(const PSet& test) {
  std::string label =  test.getParameter<std::string>("Label");
  VPSet clusterset = test.getParameter<VPSet>("Clusters");
  VPSet digiset    = test.getParameter<VPSet>("Digis");
  std::cout << "Testing: \"" << label << "\"\n";

  edmNew::DetSetVector<SiStripDigi> digis;   
  constructDigis(digiset, digis);

  output_t expected;
  constructClusters(clusterset, expected);

  output_t result;
  result.reserve(2*clusterset.size(),8*clusterset.size());
  
  try { 
    clusterizer->clusterize(digis, result); 
    assertIdentical(expected, result);
    if(test.getParameter<bool>("InvalidCharge")) throw cms::Exception("Failed") << "Charges are valid, contrary to expectation.\n";
  }
  catch(StripClusterizerAlgorithm::InvalidChargeException e) {
    if(!test.getParameter<bool>("InvalidCharge")) throw e;
  }
  catch(cms::Exception e) {
    std::cout << ( e << "Input:\n" << printDigis(digiset));
  }
}

void ClusterizerUnitTester::
constructDigis(const VPSet& stripset, edmNew::DetSetVector<SiStripDigi>& digis) {
  edmNew::DetSetVector<SiStripDigi>::FastFiller digisFF(digis, detId);
  for(iter_t strip = stripset.begin(); strip < stripset.end(); strip++) {
    digisFF.push_back( SiStripDigi(strip->getParameter<unsigned>("Strip"),
				   strip->getParameter<unsigned>("ADC") ));
  }
  if(digisFF.empty()) digisFF.abort();
}

void ClusterizerUnitTester::
constructClusters(const VPSet& clusterset, output_t& clusters) {
  output_t::FastFiller clustersFF(clusters, detId);
  for(iter_t c = clusterset.begin(); c<clusterset.end(); c++) {
    uint16_t firststrip =  c->getParameter<unsigned>("FirstStrip");
    std::vector<unsigned> amplitudes =  c->getParameter<std::vector<unsigned> >("Amplitudes");
    std::vector<uint16_t> a16(amplitudes.begin(),amplitudes.end());
    clustersFF.push_back(SiStripCluster(firststrip, a16.begin(),a16.end()));
  }
  if(clustersFF.empty()) clustersFF.abort();
}

std::string ClusterizerUnitTester::
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






void ClusterizerUnitTester::
assertIdentical(const output_t& L, const output_t& R) {
  if(!clusterDSVsIdentical(L,R))
    throw cms::Exception("Failed") << "Expected:\n" << printDSV(L) 
				   << "Actual:\n"   << printDSV(R);
}

bool ClusterizerUnitTester::
clusterDSVsIdentical(const output_t& L, const output_t& R) {
  return 
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		  bool(true), std::logical_and<bool>(), clusterDetSetsIdentical );  
}

bool ClusterizerUnitTester::
clusterDetSetsIdentical(const edmNew::DetSet<SiStripCluster>& L, const edmNew::DetSet<SiStripCluster>& R) {
  return 
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		  bool(true), std::logical_and<bool>(), clustersIdentical );
}

bool ClusterizerUnitTester::
clustersIdentical(const SiStripCluster& L, const SiStripCluster& R) {
  return
    L.firstStrip() == R.firstStrip() 
    && L.amplitudes().size() == R.amplitudes().size()
    && inner_product(L.amplitudes().begin(), L.amplitudes().end(), R.amplitudes().begin(), 
		     bool(true), std::logical_and<bool>(), std::equal_to<uint16_t>() );
}

std::string ClusterizerUnitTester::
printDSV(const output_t& dsv) {
  std::stringstream s;
  for(output_t::const_iterator 	it = dsv.begin(); it<dsv.end(); it++)
    for(edmNew::DetSet<SiStripCluster>::const_iterator cluster = it->begin(); cluster < it->end() ; cluster++)
      s << printCluster(*cluster);
  return s.str();
}

std::string ClusterizerUnitTester::
printCluster(const SiStripCluster & cluster) {
  std::stringstream s;
  s  << "\t" << cluster.firstStrip() << " [ ";
  for(unsigned i=0; i<cluster.amplitudes().size(); i++) {
    s << static_cast<int>(cluster.amplitudes()[i]) << " ";
  }
  s << "]" << std::endl;
  return s.str();
}
