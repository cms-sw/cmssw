#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <iostream>
#include <string>

class HcalGeometryPlan1Tester : public edm::one::EDAnalyzer<> {

public:
  explicit HcalGeometryPlan1Tester( const edm::ParameterSet& );
  ~HcalGeometryPlan1Tester( void ) override {}
    
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ParameterSet ps0_;
  bool              geomES_;
};

HcalGeometryPlan1Tester::HcalGeometryPlan1Tester( const edm::ParameterSet& iConfig ) :
  ps0_(iConfig) {
  geomES_ = iConfig.getParameter<bool>("GeometryFromES");
}

void HcalGeometryPlan1Tester::analyze(const edm::Event& /*iEvent*/, 
				      const edm::EventSetup& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get(topologyHandle);
  const HcalTopology topology = (*topologyHandle);
  HcalGeometry* geom(nullptr);
  if (geomES_) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
    geom = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader(ps0_);
    geom = (HcalGeometry*)(m_loader.load(topology, hcons));
  }

  std::vector<HcalDetId> idsp;
  bool ok = hcons.specialRBXHBHE(true,idsp);
  std::cout << "Special RBX Flag " << ok << " with " << idsp.size()
	    << " ID's" << std::endl;
  int nall(0), ngood(0);
  for (std::vector<HcalDetId>::const_iterator itr=idsp.begin();
       itr != idsp.end(); ++itr) {
    if (topology.valid(*itr)) {
      ++nall;
      HcalDetId idnew = hcons.mergedDepthDetId(*itr);
      GlobalPoint pt1 = geom->getGeometry(*itr)->getPosition();
      GlobalPoint pt2 = geom->getPosition(idnew);
      double     deta = pt1.eta() - pt2.eta();
      double     dphi = pt1.phi() - pt2.phi();
      ok              = (std::abs(deta)<0.00001) && (std::abs(dphi)<0.00001);
      std::cout << "Unmerged ID " << (*itr) << " (" << pt1.eta() << ", " 
		<< pt1.phi() << ", " << pt1.z() << ") Merged ID " << idnew
		<< " (" << pt2.eta() << ", " << pt2.phi() << ", " << pt2.z()
		<< ") ";
      if (ok) ++ngood;
      else    std::cout << " ***** ERROR *****";
      std::cout << std::endl;
    }
  }
  std::cout << ngood << " out of " << nall << " ID's are tested OK\n";
}

DEFINE_FWK_MODULE(HcalGeometryPlan1Tester);
