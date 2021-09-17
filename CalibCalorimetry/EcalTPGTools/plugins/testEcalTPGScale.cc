
#include "testEcalTPGScale.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

testEcalTPGScale::testEcalTPGScale(edm::ParameterSet const& pSet)
    : geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      endcapGeomToken_(
          esConsumes<CaloSubdetectorGeometry, EcalEndcapGeometryRecord>(edm::ESInputTag{"", "EcalEndcap"})),
      barrelGeomToken_(
          esConsumes<CaloSubdetectorGeometry, EcalBarrelGeometryRecord>(edm::ESInputTag{"", "EcalBarrel"})),
      eTTmapToken_(esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>()),
      tokens_(consumesCollector()) {
  std::cout << "I'm going to check the internal consistancy of EcalTPGScale transformation..." << std::endl;
}

void testEcalTPGScale::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  using namespace edm;
  using namespace std;

  // geometry
  ESHandle<CaloGeometry> theGeometry = evtSetup.getHandle(geomToken_);
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle = evtSetup.getHandle(endcapGeomToken_);
  ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle = evtSetup.getHandle(barrelGeomToken_);
  ESHandle<EcalTrigTowerConstituentsMap> eTTmap = evtSetup.getHandle(eTTmapToken_);

  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);

  EcalTPGScale ecalScale(tokens_, evtSetup);

  bool error(false);
  vector<DetId>::const_iterator it;

  // EB
  const std::vector<DetId>& ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  it = ebCells.begin();
  const EBDetId idEB(*it);
  const EcalTrigTowerDetId towidEB = idEB.tower();
  for (unsigned int ADC = 0; ADC < 256; ADC++) {
    double gev = ecalScale.getTPGInGeV(ADC, towidEB);
    unsigned int tpgADC = ecalScale.getTPGInADC(gev, towidEB);
    if (tpgADC != ADC) {
      error = true;
      cout << " ERROR : with ADC = " << ADC << " getTPGInGeV = " << gev << " getTPGInADC = " << tpgADC << endl;
    }
    ecalScale.getLinearizedTPG(ADC, towidEB);
  }

  // EE
  const std::vector<DetId>& eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  it = eeCells.begin();
  const EEDetId idEE(*it);
  const EcalTrigTowerDetId towidEE = (*eTTmap).towerOf(idEE);
  for (unsigned int ADC = 0; ADC < 256; ADC++) {
    double gev = ecalScale.getTPGInGeV(ADC, towidEE);
    unsigned int tpgADC = ecalScale.getTPGInADC(gev, towidEE);
    if (tpgADC != ADC) {
      error = true;
      cout << " ERROR : with ADC = " << ADC << " getTPGInGeV = " << gev << " getTPGInADC = " << tpgADC << endl;
    }
    ecalScale.getLinearizedTPG(ADC, towidEE);
  }

  if (!error)
    cout << " there is no error with EcalTPGScale internal consistancy " << endl;
}

void testEcalTPGScale::beginJob() {
  using namespace edm;
  using namespace std;
}
