#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


struct TestEgammaTowerIsolation : public edm::EDAnalyzer {

 explicit  TestEgammaTowerIsolation(const edm::ParameterSet&){}
  ~TestEgammaTowerIsolation(){}
  
  //  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob(){}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}


  std::string towerLabel="towerMaker";



};
//define this as a plug-in
DEFINE_FWK_MODULE(TestEgammaTowerIsolation);


#include <iostream>
void TestEgammaTowerIsolation::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<CaloTowerCollection> towerHandle;
  iEvent.getByLabel(towerLabel, towerHandle);
  const CaloTowerCollection & towers = *towerHandle.product();
  
  std::cout << "\n new event\n " << towers.size() << std::endl;

  uint32_t nt = towers.size();
  for ( uint32_t j=0;j!=nt; ++j) {
    std:: cout << j << ": " << 
      towers[j].eta() << ", " <<
      towers[j].phi() << ", " <<
      towers[j].id() << ", " <<
      towers[j].ietaAbs() << ", " <<
      std::sin(towers[j].theta()) << ", " <<
      1./std::cosh(towers[j].eta()) << ", " <<
      towers[j].hadEnergy() << ", " <<
      towers[j].hadEnergyHeInnerLayer() << ", " <<
      towers[j].hadEnergyHeOuterLayer() << ", " <<
      std::endl;
  }



}
