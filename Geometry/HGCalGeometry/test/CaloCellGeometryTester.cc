#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <iostream>
#include <string>

class CaloCellGeometryTester : public edm::one::EDAnalyzer<> {

public:
  explicit CaloCellGeometryTester( const edm::ParameterSet& );
  ~CaloCellGeometryTester( void ) override {}
    
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}
};

CaloCellGeometryTester::CaloCellGeometryTester(const edm::ParameterSet&) { }

void CaloCellGeometryTester::analyze(const edm::Event& /*iEvent*/, 
				     const edm::EventSetup& iSetup) {


  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  int         dets[9] = {3,3,3,4,4,4,4,6,6};
  int         subd[9] = {1,2,3,1,2,4,3,3,4};
  std::string name[9] = {"EB","EE","ES","HB","HE","HF","HO","HGCEE","HGCFH"};
  for (unsigned int k=0; k<9; ++k) {
    const CaloSubdetectorGeometry* geom = geo->getSubdetectorGeometry((DetId::Detector)(dets[k]),subd[k]);
    if (geom) {
      std::cout << name[k] << " has " << geom->getValidDetIds((DetId::Detector)(dets[k]),subd[k]).size() << " valid cells" << std::endl;
      if (k == 7 || k == 8)
	std::cout << "Number of valid GeomID " 
		  << ((HGCalGeometry*)(geom))->getValidGeomDetIds().size()
		  << std::endl;
    }
  }

  std::string named1[2] = {"FastTimeBarrel","SFBX"};
  std::string named2[2] = {"FTBarrel","FTEndcap"};
  for (int k=0; k<2; ++k) {
    edm::ESHandle<FastTimeGeometry> fgeom;
    iSetup.get<IdealGeometryRecord>().get(named1[k],fgeom);
    if (fgeom.isValid()) {
      const FastTimeGeometry* geom = (fgeom.product());
      std::cout << named2[k] << " has " << geom->getValidDetIds().size() << " valid cells" << std::endl;
      std::cout << "Number of valid GeomID " 
		<< geom->getValidGeomDetIds().size() << std::endl;
    }
  }
}

DEFINE_FWK_MODULE(CaloCellGeometryTester);
