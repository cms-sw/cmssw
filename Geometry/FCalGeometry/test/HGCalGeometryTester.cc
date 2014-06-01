#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "CoralBase/Exception.h"

class HGCalGeometryTester : public edm::EDAnalyzer {
public:
  explicit HGCalGeometryTester(const edm::ParameterSet& );
  ~HGCalGeometryTester();

  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void doTest(const HGCalGeometry& geom);

private:
  // ----------member data ---------------------------
};

HGCalGeometryTester::HGCalGeometryTester(const edm::ParameterSet& ) {}


HGCalGeometryTester::~HGCalGeometryTester() {}

void HGCalGeometryTester::analyze(const edm::Event& , 
				  const edm::EventSetup& iSetup ) {

  std::string name;
  edm::ESHandle<HGCalGeometry> geom;

  name = "HGCalEESensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) doTest(*geom);
  else                std::cout << "Cannot get valid HGCalGeometry Object for "
				<< name << std::endl;

  name = "HGCalHESiliconSensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) doTest(*geom);
  else                std::cout << "Cannot get valid HGCalGeometry Object for "
				<< name << std::endl;

  name = "HGCalHEScintillatorSensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) doTest(*geom);
  else                std::cout << "Cannot get valid HGCalGeometry Object for "
				<< name << std::endl;
}

void HGCalGeometryTester::doTest(const HGCalGeometry& geom) {
  
  const std::vector<DetId>& ids = geom.getValidDetIds();
  std::cout << ids.size() << " valid ids for " << geom.cellElement() 
	    << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryTester);
