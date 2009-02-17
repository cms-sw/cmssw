#ifndef ClusterizerUnitTester_h
#define ClusterizerUnitTester_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

class ClusterizerUnitTester : public edm::EDAnalyzer {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;
public:
  ClusterizerUnitTester(const PSet& conf) :
    testGroups(conf.getParameter<VPSet>("ClusterizerTestGroups")),
    clusterizer(0),
    detId(0) {}
  ~ClusterizerUnitTester() { if(clusterizer) delete clusterizer; clusterizer=0;}
private:
  void analyze(const edm::Event&, const edm::EventSetup&);
  
  void initializeTheGroup(const PSet&, const edm::EventSetup&);
  void testTheGroup(const PSet&);
  void runTheTest(const PSet&);

  void constructClusters(const VPSet&, edmNew::DetSetVector<SiStripCluster>::FastFiller&);
  static void constructDigis(const VPSet&, edm::DetSet<SiStripDigi>&);
  static void assertIdentical(const edmNew::DetSetVector<SiStripCluster>&, 
			      const edmNew::DetSetVector<SiStripCluster>&);
  static bool clusterDSVsIdentical( const edmNew::DetSetVector<SiStripCluster>&, 
				    const edmNew::DetSetVector<SiStripCluster>&);
  static bool clusterDetSetsIdentical( const edmNew::DetSet<SiStripCluster>&, 
				       const edmNew::DetSet<SiStripCluster>&);
  static bool clustersIdentical(const SiStripCluster&, const SiStripCluster&);
  static std::string printDSV(const edmNew::DetSetVector<SiStripCluster>&) {return "";}

  VPSet testGroups;
  ThreeThresholdStripClusterizer* clusterizer;
  uint32_t detId;
};

#endif
