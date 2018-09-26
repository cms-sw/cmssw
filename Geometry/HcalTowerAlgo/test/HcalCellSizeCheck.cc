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
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include <iostream>
#include <string>

class HcalCellSizeCheck : public edm::one::EDAnalyzer<> {

public:
  explicit HcalCellSizeCheck(const edm::ParameterSet& );
  ~HcalCellSizeCheck( void ) override {}

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}
};

HcalCellSizeCheck::HcalCellSizeCheck( const edm::ParameterSet& iConfig ) { }

void HcalCellSizeCheck::analyze(const edm::Event& /*iEvent*/,
				 const edm::EventSetup& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);

  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get(topologyHandle);
  const HcalTopology topology = (*topologyHandle);

  HcalFlexiHardcodeGeometryLoader m_loader;
  CaloSubdetectorGeometry*  geom = m_loader.load(topology, hcons);

  const std::vector<DetId>& idsb=geom->getValidDetIds(DetId::Hcal,HcalBarrel);
  for (auto id : idsb) {
    HcalDetId hid(id.rawId());
    std::pair<double,double> rz = hcons.getRZ(hid);
    std::cout << hid << " Front " << rz.first << " Back " << rz.second << "\n";
  }

  const std::vector<DetId>& idse=geom->getValidDetIds(DetId::Hcal,HcalEndcap);
  for (auto id : idse) {
    HcalDetId hid(id.rawId());
    std::pair<double,double> rz = hcons.getRZ(hid);
    std::cout << hid << " Front " << rz.first << " Back " << rz.second << "\n";
  }
}

DEFINE_FWK_MODULE(HcalCellSizeCheck);
