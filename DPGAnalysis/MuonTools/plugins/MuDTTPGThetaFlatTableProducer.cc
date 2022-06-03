/** \class MuDTTPGThetaFlatTableProducer MuDTTPGThetaFlatTableProducer.cc DPGAnalysis/MuonTools/plugins/MuDTTPGThetaFlatTableProducer.cc
 *  
 * Helper class : the Phase-1 local trigger FlatTableProducer for TwinMux in/out and BMTF in (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/plugins/MuDTTPGThetaFlatTableProducer.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include <iostream>
#include <vector>

MuDTTPGThetaFlatTableProducer::MuDTTPGThetaFlatTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer{config}, m_tag{getTag(config)}, m_token{config, consumesCollector(), "src"} {
  produces<nanoaod::FlatTable>();
}

void MuDTTPGThetaFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "ltBmtfInTh");
  desc.ifValue(edm::ParameterDescription<std::string>("tag", "BMTF_IN", true),
               edm::allowedValues<std::string>("BMTF_IN", "TM_IN"));
  desc.add<edm::InputTag>("src", edm::InputTag{"bmtfDigis"});

  descriptions.addWithDefaultLabel(desc);
}

void MuDTTPGThetaFlatTableProducer::fillTable(edm::Event& ev) {
  unsigned int nTrigs{0};

  std::vector<int8_t> wheel;
  std::vector<int8_t> sector;
  std::vector<int8_t> station;

  std::vector<int8_t> bx;
  std::vector<uint32_t> hitMap;

  auto trigColl = m_token.conditionalGet(ev);

  if (trigColl.isValid()) {
    const auto trigs = trigColl->getContainer();
    for (const auto& trig : (*trigs)) {
      bool hasData = false;
      for (int pos = 0; pos < 7; ++pos) {
        if (trig.code(pos)) {
          hasData = true;
          break;
        }
      }

      if (!hasData)
        continue;

      wheel.push_back(trig.whNum());
      sector.push_back(trig.scNum() + (m_tag != TriggerTag::BMTF_IN ? 1 : 0));
      station.push_back(trig.stNum());

      bx.push_back(trig.bxNum());

      uint32_t hitMapCh = 0;
      for (int pos = 0; pos < 7; ++pos)
        if (trig.code(pos))
          hitMapCh = hitMapCh | (0x1 << pos);

      hitMap.push_back(hitMapCh);

      ++nTrigs;
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nTrigs, m_name, false, false);

  table->setDoc("Barrel trigger primitive information (theta view)");

  addColumn(table, "wheel", wheel, "wheel - [-2:2] range");
  addColumn(table,
            "sector",
            sector,
            "sector"
            "<br /> - [1:12] range for TwinMux"
            "<br /> - [0:11] range for BMTF input"
            "<br /> - double MB4 stations are part of S4 and S10 in TwinMux"
            "<br /> - double MB4 stations are part of S3 and S9 in BMTF input");
  addColumn(table, "station", station, "station - [1:3] range");
  addColumn(table,
            "bx",
            bx,
            "bx:"
            "<br /> - BX = 0 is the one where the event is collected"
            "<br /> - TwinMux range [X:Y]"
            "<br /> - BMTF input range [X:Y]");
  addColumn(table,
            "hitMap",
            hitMap,
            "Map groups of BTIs that fired (unsigned int):"
            "<br /> there are 7 groups of BTI per chamber, the first one"
            "<br /> being the less significant bit of the map [CHECK]");

  ev.put(std::move(table));
}

MuDTTPGThetaFlatTableProducer::TriggerTag MuDTTPGThetaFlatTableProducer::getTag(const edm::ParameterSet& config) {
  auto tag{TriggerTag::TM_IN};

  auto tagName = config.getParameter<std::string>("tag");

  if (tagName != "TM_IN" && tagName != "BMTF_IN")
    edm::LogError("") << "[MuDTTPGThetaFlatTableProducer]::getTag: " << tagName
                      << " is not a valid tag, defaulting to TM_IN";

  if (tagName == "BMTF_IN")
    tag = TriggerTag::BMTF_IN;

  return tag;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuDTTPGThetaFlatTableProducer);