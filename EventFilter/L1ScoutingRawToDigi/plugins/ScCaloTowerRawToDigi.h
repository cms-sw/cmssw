#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"

#include "EventFilter/L1ScoutingRawToDigi/interface/shifts.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"
#include "L1TriggerScouting/Utilities/interface/printScObjects.h"

#include <memory>
#include <vector>
#include <iostream>

class ScCaloTowerRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScCaloTowerRawToDigi(const edm::ParameterSet&);
  ~ScCaloTowerRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void unpackOrbit(const unsigned char* buf, size_t len, int sdsId);

  // vector holding data for every bunch crossing
  // before filling the orbit collection
  std::vector<std::vector<l1ScoutingRun3::CaloTower>> orbitBuffer_;
  int nCaloTowersOrbit_;

  bool debug_ = false;
  std::vector<int> sourceIdList_;
  edm::InputTag srcInputTag_;
  edm::EDGetToken rawToken_;
};
