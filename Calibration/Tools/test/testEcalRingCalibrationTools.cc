// -*- C++ -*-
//
// Package:    testEcalRingCalibrationTools
// Class:      testEcalRingCalibrationTools
//
/**\class testEcalRingCalibrationTools testEcalRingCalibrationTools.cc test/testEcalRingCalibrationTools/src/testEcalRingCalibrationTools.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

// system include files
#include <memory>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include <fstream>

//
// class decleration
//

class testEcalRingCalibrationTools : public edm::EDAnalyzer {
public:
  explicit testEcalRingCalibrationTools(const edm::ParameterSet&);
  ~testEcalRingCalibrationTools() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  void build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name);
  int pass_;
  //  bool fullEcalDump_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
testEcalRingCalibrationTools::testEcalRingCalibrationTools(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  pass_ = 0;
  //  fullEcalDump_=iConfig.getUntrackedParameter<bool>("fullEcalDump",false);
}

testEcalRingCalibrationTools::~testEcalRingCalibrationTools() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void testEcalRingCalibrationTools::build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name) {
  std::fstream f(name, std::ios_base::out);
  const CaloSubdetectorGeometry* geom = cg.getSubdetectorGeometry(det, subdetn);

  const std::vector<DetId>& ids = geom->getValidDetIds(det, subdetn);
  if (det == DetId::Ecal && subdetn == EcalBarrel) {
    f << "EB-" << std::endl;
    for (int iphi = 360; iphi > 0; --iphi) {
      for (int ieta = -85; ieta <= -1; ++ieta) {
        if (EBDetId::validDetId(ieta, iphi))
          f << std::setw(4) << EcalRingCalibrationTools::getRingIndex(EBDetId(ieta, iphi));
        else
          f << "XXXX";
      }
      f << std::endl;
    }

    for (unsigned int i = 0; i < 10; i++)
      f << std::endl;

    f << "EB+" << std::endl;
    for (int iphi = 360; iphi > 0; --iphi) {
      for (int ieta = 1; ieta <= 85; ++ieta) {
        if (EBDetId::validDetId(ieta, iphi))
          f << std::setw(4) << EcalRingCalibrationTools::getRingIndex(EBDetId(ieta, iphi));
        else
          f << "XXXX";
      }
      f << std::endl;
    }

    for (int i = 0; i < EcalRingCalibrationTools::N_RING_BARREL; ++i) {
      f << "++++++++++Ring Index " << i << " +++++++++++++++++" << std::endl;
      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(i);
      std::cout << i << " " << ringIds.size() << std::endl;
      assert(ringIds.size() == 360);
      for (unsigned int iid = 0; iid < ringIds.size(); ++iid)
        f << EBDetId(ringIds[iid]) << std::endl;
    }
  }

  if (det == DetId::Ecal && subdetn == EcalEndcap) {
    f << "EE-" << std::endl;
    for (int iy = 100; iy > 0; --iy) {
      for (int ix = 1; ix <= EEDetId::IX_MAX; ++ix) {
        if (EEDetId::validDetId(ix, iy, -1))
          f << std::setw(4) << EcalRingCalibrationTools::getRingIndex(EEDetId(ix, iy, -1));
        else
          f << "XXXX";
      }
      f << std::endl;
    }

    for (unsigned int i = 0; i < 10; i++)
      f << std::endl;

    f << "EE+" << std::endl;
    for (int iy = 100; iy > 0; --iy) {
      for (int ix = 1; ix <= EEDetId::IX_MAX; ++ix) {
        if (EEDetId::validDetId(ix, iy, 1))
          f << std::setw(4) << EcalRingCalibrationTools::getRingIndex(EEDetId(ix, iy, +1));
        else
          f << "XXXX";
      }
      f << std::endl;
    }

    unsigned int totalRingSize = 0;
    for (int i = EcalRingCalibrationTools::N_RING_BARREL; i < EcalRingCalibrationTools::N_RING_TOTAL; ++i) {
      f << "++++++++++Ring Index " << i << " +++++++++++++++++" << std::endl;
      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(i);
      std::cout << i << " " << ringIds.size() << std::endl;
      totalRingSize += ringIds.size();
      for (unsigned int iid = 0; iid < ringIds.size(); ++iid)
        f << EEDetId(ringIds[iid]) << std::endl;
    }
    assert(totalRingSize == ids.size());
  }
  f.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void testEcalRingCalibrationTools::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::cout << "Here I am " << std::endl;

  // get the ecal & hcal geometry
  //
  if (pass_ == 1) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    EcalRingCalibrationTools::setCaloGeometry(&(*pG));
    build(*pG, DetId::Ecal, EcalBarrel, "eb.ringDump");
    build(*pG, DetId::Ecal, EcalEndcap, "ee.ringDump");
  }

  pass_++;
}

//define this as a plug-in

DEFINE_FWK_MODULE(testEcalRingCalibrationTools);
