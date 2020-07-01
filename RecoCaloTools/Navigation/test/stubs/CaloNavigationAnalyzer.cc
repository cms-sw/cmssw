// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include <iostream>

class CaloNavigationAnalyzer : public edm::EDAnalyzer {
public:
  explicit CaloNavigationAnalyzer(const edm::ParameterSet&);
  ~CaloNavigationAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  int pass_;
};

//
// constructors and destructor
//
CaloNavigationAnalyzer::CaloNavigationAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  pass_ = 0;
}

CaloNavigationAnalyzer::~CaloNavigationAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CaloNavigationAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  if (pass_ == 0) {
    {
      std::cout << "Testing Ecal Barrel Navigator" << std::endl;
      EBDetId startEB(-85, 1);
      std::cout << "Starting at : (" << startEB.ieta() << "," << startEB.iphi() << ")" << std::endl;
      EcalBarrelNavigator theEBNav(startEB, theCaloTopology->getSubdetectorTopology(DetId::Ecal, EcalBarrel));

      EBDetId next;
      int steps = 0;
      while (((next = theEBNav.north()) != EBDetId(0) && next != startEB)) {
        std::cout << "North " << steps << " : (" << next.ieta() << "," << next.iphi() << ")" << std::endl;
        ++steps;
      }
      theEBNav.home();
      steps = 0;
      while (((next = theEBNav.south()) != EBDetId(0) && next != startEB)) {
        std::cout << "South " << steps << " : (" << next.ieta() << "," << next.iphi() << ")" << std::endl;
        ++steps;
      }
      theEBNav.home();
      steps = 0;
      while (((next = theEBNav.west()) != EBDetId(0) && next != startEB)) {
        std::cout << "West " << steps << " : (" << next.ieta() << "," << next.iphi() << ")" << std::endl;
        ++steps;
      }
      theEBNav.home();
      steps = 0;
      while (((next = theEBNav.east()) != EBDetId(0) && next != startEB)) {
        std::cout << "East " << steps << " : (" << next.ieta() << "," << next.iphi() << ")" << std::endl;
        ++steps;
      }
      theEBNav.home();
      std::cout << "Coming back home" << std::endl;
    }
    {
      std::cout << "Testing Ecal Endcap Navigator" << std::endl;
      EEDetId startEE(1, 50, 1);
      std::cout << "Starting at : (" << startEE.ix() << "," << startEE.iy() << ")" << std::endl;
      EcalEndcapNavigator theEENav(startEE, theCaloTopology->getSubdetectorTopology(DetId::Ecal, EcalEndcap));
      theEENav.setHome(startEE);

      EEDetId next;
      int steps = 0;
      while (((next = theEENav.north()) != EEDetId(0) && next != startEE)) {
        std::cout << "North " << steps << " : (" << next.ix() << "," << next.iy() << ")" << std::endl;
        ++steps;
      }
      theEENav.home();
      steps = 0;
      while (((next = theEENav.south()) != EEDetId(0) && next != startEE)) {
        std::cout << "South " << steps << " : (" << next.ix() << "," << next.iy() << ")" << std::endl;
        ++steps;
      }
      theEENav.home();
      steps = 0;
      while (((next = theEENav.west()) != EEDetId(0) && next != startEE)) {
        std::cout << "West " << steps << " : (" << next.ix() << "," << next.iy() << ")" << std::endl;
        ++steps;
      }
      theEENav.home();
      steps = 0;
      while (((next = theEENav.east()) != EEDetId(0) && next != startEE)) {
        std::cout << "East " << steps << " : (" << next.ix() << "," << next.iy() << ")" << std::endl;
        ++steps;
      }
      theEENav.home();
    }
    {
      std::cout << "Testing Ecal Preshower Navigator" << std::endl;
      ESDetId startES(1, 16, 1, 1, 1);
      std::cout << "Starting at : " << startES << std::endl;
      EcalPreshowerNavigator theESNav(startES, theCaloTopology->getSubdetectorTopology(DetId::Ecal, EcalPreshower));
      theESNav.setHome(startES);

      ESDetId next;
      int steps = 0;
      while (((next = theESNav.north()) != ESDetId(0) && next != startES)) {
        std::cout << "North " << steps << " : " << next << std::endl;
        ++steps;
      }
      theESNav.home();
      steps = 0;
      while (((next = theESNav.south()) != ESDetId(0) && next != startES)) {
        std::cout << "South " << steps << " : " << next << std::endl;
        ++steps;
      }
      theESNav.home();
      steps = 0;
      while (((next = theESNav.west()) != ESDetId(0) && next != startES)) {
        std::cout << "West " << steps << " : " << next << std::endl;
        ++steps;
      }
      theESNav.home();
      steps = 0;
      while (((next = theESNav.east()) != ESDetId(0) && next != startES)) {
        std::cout << "East " << steps << " : " << next << std::endl;
        ++steps;
      }
      theESNav.home();
    }
  }

  pass_++;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloNavigationAnalyzer);
