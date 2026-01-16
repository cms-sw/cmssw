/** \class MuGEML1FETableProducer MuGEML1FETableProducer.cc DPGAnalysis/MuonTools/src/MuGEML1FETableProducer.cc
 *  
 * Helper class : FlatTableProducer for GEM Flower Event (reading FED RAW Data)
 *
 * \author Jeewon Heo
 * based on code written by C.Battilana (INFN BO)
 *
 */

//#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <vector>

#include "DPGAnalysis/MuonTools/interface/MuBaseFlatTableProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"

class MuGEML1FETableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuGEML1FETableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given event
  void fillTable(edm::Event&) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run&, const edm::EventSetup&) final;

private:
  nano_mu::EDTokenHandle<TCDSRecord> m_token;
  static constexpr int BX_IN_ORBIT = 3564;
};

MuGEML1FETableProducer::MuGEML1FETableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer{config}, m_token{config, consumesCollector(), "src"} {
  produces<nanoaod::FlatTable>();
}

void MuGEML1FETableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "l1aHistory");
  desc.add<edm::InputTag>("src", edm::InputTag{"tcdsDigis:tcdsRecord"});

  descriptions.addWithDefaultLabel(desc);
}

void MuGEML1FETableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {}

void MuGEML1FETableProducer::fillTable(edm::Event& ev) {
  std::vector<int> bxDiffs;

  auto record = m_token.conditionalGet(ev);

  // in Heavy Ion Physics the getL1aHistoryEntry is not saved ...
  // comment out and use this as proxy to inquire BX,Orbit and Lumi
  for (const auto l1aEntry : record->getFullL1aHistory()) {
    int bxDiff = BX_IN_ORBIT * (record->getOrbitNr() - l1aEntry.getOrbitNr()) + record->getBXID() - l1aEntry.getBXID();
    bxDiffs.push_back(bxDiff);
  }

  auto table = std::make_unique<nanoaod::FlatTable>(bxDiffs.size(), m_name, false, false);
  addColumn(table, "bxDiffs", bxDiffs, "BX differences between event and L1As");

  ev.put(std::move(table));
}

DEFINE_FWK_MODULE(MuGEML1FETableProducer);
