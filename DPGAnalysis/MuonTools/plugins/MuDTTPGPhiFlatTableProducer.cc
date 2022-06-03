/** \class MuDTTPGPhiFlatTableProducer MuDTTPGPhiFlatTableProducer.cc DPGAnalysis/MuonTools/src/MuDTTPGPhiFlatTableProducer.cc
 *  
 * Helper class : the Phase-1 local trigger FlatTableProducer for TwinMux in/out and BMTF in (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/plugins/MuDTTPGPhiFlatTableProducer.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include <iostream>
#include <vector>

MuDTTPGPhiFlatTableProducer::MuDTTPGPhiFlatTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer{config},
      m_tag{getTag(config)},
      m_token{config, consumesCollector(), "src"},
      m_trigGeomUtils{consumesCollector()} {
  produces<nanoaod::FlatTable>();
}

void MuDTTPGPhiFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "ltBmtfIn");
  desc.ifValue(edm::ParameterDescription<std::string>("tag", "BMTF_IN", true),
               edm::allowedValues<std::string>("BMTF_IN", "TM_IN", "TM_OUT"));
  desc.add<edm::InputTag>("src", edm::InputTag{"bmtfDigis"});

  descriptions.addWithDefaultLabel(desc);
}

void MuDTTPGPhiFlatTableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {
  m_trigGeomUtils.getFromES(run, environment);
}

void MuDTTPGPhiFlatTableProducer::fillTable(edm::Event& ev) {
  unsigned int nTrigs{0};

  std::vector<int8_t> wheel;
  std::vector<int8_t> sector;
  std::vector<int8_t> station;

  std::vector<int8_t> quality;
  std::vector<int8_t> rpcBit;

  std::vector<int> phi;
  std::vector<int> phiB;

  std::vector<float> posLoc_x;
  std::vector<float> dirLoc_phi;

  std::vector<int8_t> bx;
  std::vector<int8_t> is2nd;

  auto trigColl = m_token.conditionalGet(ev);

  if (trigColl.isValid()) {
    const auto trigs = trigColl->getContainer();
    for (const auto& trig : (*trigs)) {
      if (trig.code() != 7) {
        wheel.push_back(trig.whNum());
        sector.push_back(trig.scNum() + (m_tag != TriggerTag::BMTF_IN ? 1 : 0));
        station.push_back(trig.stNum());

        quality.push_back(trig.code());

        if (m_tag == TriggerTag::TM_OUT)
          rpcBit.push_back(trig.RpcBit());

        phi.push_back(trig.phi());
        phiB.push_back(trig.phiB());

        auto [x, dir] = m_trigGeomUtils.trigToReco(&trig);

        posLoc_x.push_back(x);
        dirLoc_phi.push_back(dir);

        bx.push_back(trig.bxNum() - (m_tag == TriggerTag::TM_IN && trig.Ts2Tag() ? 1 : 0));
        is2nd.push_back(trig.Ts2Tag());

        ++nTrigs;
      }
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nTrigs, m_name, false, false);

  table->setDoc("Barrel trigger primitive information (phi view)");

  addColumn(table, "wheel", wheel, "wheel - [-2:2] range");
  addColumn(table,
            "sector",
            sector,
            "sector"
            "<br /> - [1:12] range for TwinMux"
            "<br /> - [0:11] range for BMTF input"
            "<br /> - double MB4 stations are part of S4 and S10 in TwinMux"
            "<br /> - double MB4 stations are part of S3 and S9 in BMTF input");
  addColumn(table, "station", station, "station - [1:4] range");
  addColumn(table,
            "quality",
            quality,
            "quality - [0:6] range"
            "<br /> - [0:1] : uncorrelated L triggers"
            "<br /> - [2:3] : uncorrelated H triggers"
            "<br /> - 4 : correlated LL triggers"
            "<br /> - 5 : correlated HL triggers"
            "<br /> - 6 : correlated HH triggers");
  if (m_tag == TriggerTag::TM_OUT) {
    addColumn(table,
              "rpcBit",
              rpcBit,
              "use of RPC - [0:2] range"
              "<br /> - 0 : RPC not used"
              "<br /> - 1 : RPC+DT combined trigger"
              "<br /> - 2 : RPC-only trigger");
  }

  addColumn(table,
            "phi",
            phi,
            "phi - scale and range:"
            "<br /> - 4096 correstpond to 1 rad"
            "<br /> - 0 is @ (DT sector - 1) * 30 deg in global CMS phi");
  addColumn(table,
            "phiB",
            phiB,
            "phiB - scale and range:"
            "<br /> - 512 correstpond to 1 rad"
            "<br /> - 0 is a muon with infinite pT (straight line)");
  addColumn(table, "posLoc_x", posLoc_x, "position x in chamber local coordinates - cm");
  addColumn(table, "dirLoc_phi", dirLoc_phi, "direction phi angle in chamber local coordinates - deg");
  addColumn(table,
            "bx",
            bx,
            "bx:"
            "<br /> - BX = 0 is the one where the event is collected"
            "<br /> - TwinMux range [X:Y]"
            "<br /> - BMT input range [X:Y]");
  addColumn(table, "is2nd", is2nd, "1st/2nd track flag - [0:1]");

  ev.put(std::move(table));
}

MuDTTPGPhiFlatTableProducer::TriggerTag MuDTTPGPhiFlatTableProducer::getTag(const edm::ParameterSet& config) {
  auto tag{TriggerTag::TM_IN};

  auto tagName = config.getParameter<std::string>("tag");

  if (tagName != "TM_IN" && tagName != "TM_OUT" && tagName != "BMTF_IN")
    edm::LogError("") << "[MuDTTPGPhiFlatTableProducer]::getTag: " << tagName
                      << " is not a valid tag, defaulting to TM_IN";

  if (tagName == "TM_OUT") {
    tag = TriggerTag::TM_OUT;
  } else if (tagName == "BMTF_IN") {
    tag = TriggerTag::BMTF_IN;
  }

  return tag;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuDTTPGPhiFlatTableProducer);
