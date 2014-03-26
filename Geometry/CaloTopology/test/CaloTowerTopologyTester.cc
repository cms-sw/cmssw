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

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "CoralBase/Exception.h"

class CaloTowerTopologyTester : public edm::EDAnalyzer {
public:
  explicit CaloTowerTopologyTester(const edm::ParameterSet& );
  ~CaloTowerTopologyTester();

  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void doTest(const CaloTowerTopology& topology);

private:
  // ----------member data ---------------------------
};

CaloTowerTopologyTester::CaloTowerTopologyTester(const edm::ParameterSet& ) {}


CaloTowerTopologyTester::~CaloTowerTopologyTester() {}

void CaloTowerTopologyTester::analyze(const edm::Event& , 
				 const edm::EventSetup& iSetup ) {
  edm::ESHandle<CaloTowerTopology> topo;
  iSetup.get<HcalRecNumberingRecord>().get(topo);
  if (topo.isValid()) doTest(*topo);
  else                std::cout << "Cannot get a valid CaloTowerTopology Object\n";
}

void CaloTowerTopologyTester::doTest(const CaloTowerTopology& topology) {

  for (int ieta=-topology.lastHFRing(); ieta<=topology.lastHFRing(); ieta++) {
    for (int iphi=1; iphi<=72; iphi++) {
	  const CaloTowerDetId id(ieta,iphi);
      if (topology.validDetId(id)) {  
        std::vector<DetId> idE = topology.east(id);
        std::vector<DetId> idW = topology.west(id);
        std::vector<DetId> idN = topology.north(id);
        std::vector<DetId> idS = topology.south(id);
        std::cout << "Neighbours for : Tower " << id << std::endl;
        std::cout << "          " << idE.size() << " sets along East:";
        for (unsigned int i=0; i<idE.size(); ++i) 
          std::cout << " " << (CaloTowerDetId)(idE[i]());
        std::cout << std::endl;
        std::cout << "          " << idW.size() << " sets along West:";
        for (unsigned int i=0; i<idW.size(); ++i) 
          std::cout << " " << (CaloTowerDetId)(idW[i]());
        std::cout << std::endl;
        std::cout << "          " << idN.size() << " sets along North:";
        for (unsigned int i=0; i<idN.size(); ++i) 
          std::cout << " " << (CaloTowerDetId)(idN[i]());
        std::cout << std::endl;
        std::cout << "          " << idS.size() << " sets along South:";
        for (unsigned int i=0; i<idS.size(); ++i) 
          std::cout << " " << (CaloTowerDetId)(idS[i]());
        std::cout << std::endl;
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowerTopologyTester);
