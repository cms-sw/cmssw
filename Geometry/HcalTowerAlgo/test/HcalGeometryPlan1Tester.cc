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
  bool              geomES_;
};

HcalGeometryPlan1Tester::HcalGeometryPlan1Tester( const edm::ParameterSet& iConfig ) {
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
  //  HcalGeometry* geom(nullptr);
  const CaloSubdetectorGeometry* geom(nullptr);
  if (geomES_) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
    geom = (geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    geom = (m_loader.load(topology, hcons));
  }
  //  geom  = (HcalGeometry*)(geom0);

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
      GlobalPoint pt1 = (dynamic_cast<const HcalGeometry*>(geom))->getGeometryBase(*itr)->getPosition();
      auto        ptr = geom->getGeometry(idnew);
      GlobalPoint pt2 = ptr->getPosition();
      GlobalPoint pt0 = (dynamic_cast<const HcalGeometry*>(geom))->getPosition(idnew);
      double     deta = std::abs(pt1.eta() - pt2.eta());
      double     dphi = std::abs(pt1.phi() - pt2.phi());
      if (dphi > M_PI) dphi -= (2*M_PI);
      ok              = (deta<0.00001) && (dphi<0.00001);
      deta = std::abs(pt0.eta() - pt2.eta());
      dphi = std::abs(pt0.phi() - pt2.phi());
      if (dphi > M_PI) dphi -= (2*M_PI);
      if ((deta>0.00001) || (dphi>0.00001)) ok = false;
      std::cout << "Unmerged ID " << (*itr) << " (" << pt1.eta() << ", "
		<< pt1.phi() << ", " << pt1.z() << ") Merged ID " << idnew
		<< " (" << pt2.eta() << ", " << pt2.phi() << ", " << pt2.z()
		<< ") or (" << pt0.eta() << ", " << pt0.phi() << ", "
		<< pt0.z() << ")";
      if (ok) ++ngood;
      else    std::cout << " ***** ERROR *****";
      std::cout << std::endl;
    }
  }
  std::cout << ngood << " out of " << nall << " ID's are tested OK\n";
}

DEFINE_FWK_MODULE(HcalGeometryPlan1Tester);
