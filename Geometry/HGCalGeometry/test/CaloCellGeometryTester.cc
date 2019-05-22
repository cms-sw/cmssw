#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
    
  void analyze(edm::Event const&, edm::EventSetup const&) override;
private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloToken_;
  std::array<edm::ESGetToken<FastTimeGeometry, IdealGeometryRecord>, 2> fastTimeTokens_;
};

CaloCellGeometryTester::CaloCellGeometryTester(const edm::ParameterSet&):
  caloToken_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})},
  fastTimeTokens_{{esConsumes<FastTimeGeometry, IdealGeometryRecord>(edm::ESInputTag{"", "FastTimeBarrel"}),
                   esConsumes<FastTimeGeometry, IdealGeometryRecord>(edm::ESInputTag{"", "SFBX"})}}
 { }

void CaloCellGeometryTester::analyze(const edm::Event& /*iEvent*/, 
				     const edm::EventSetup& iSetup) {


  // get handles to calogeometry and calotopology
  const auto& pG = iSetup.getData(caloToken_);
  const CaloGeometry* geo = &pG;

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

  std::string named[2] = {"FTBarrel","FTEndcap"};
  for (int k=0; k<2; ++k) {
    if (auto fgeom = iSetup.getHandle(fastTimeTokens_[k])) {
      const FastTimeGeometry* geom = (fgeom.product());
      std::cout << named[k] << " has " << geom->getValidDetIds().size() << " valid cells" << std::endl;
      std::cout << "Number of valid GeomID " 
		<< geom->getValidGeomDetIds().size() << std::endl;
    }
  }
}

DEFINE_FWK_MODULE(CaloCellGeometryTester);
