// -*- C++ -*-
//
// Package:     L1TriggerScouting/Utilities
// Class  :     OrbitNanoAODOutputModule
//
// Implementation:
//     Adapt from NanoAODOutputModule for OrbitFlatTable
//     This handles rotating OrbitCollection to Event
//
//
// Author original version: Giovanni Petrucciani
//        adapted by Patin Inkaew
//
#include <vector>

#include "oneapi/tbb/task_arena.h"

#include "DataFormats/NanoAOD/interface/OrbitFlatTable.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/NanoAOD/interface/NanoAODOutputModuleBase.h"

#include "OrbitTableOutputBranches.h"
#include "SelectedBxTableOutputBranches.h"

class OrbitNanoAODOutputModule : public NanoAODOutputModuleBase {
public:
  OrbitNanoAODOutputModule(edm::ParameterSet const& pset);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void writeEventTree(edm::EventForOutput const& iEvent) override;
  void writeLuminosityBlockTree(edm::LuminosityBlockForOutput const& iLumi) override;
  void writeRunTree(edm::RunForOutput const& iRun) override;

  void initTables() override;
  void initEventTree(TTree& tree) override { m_commonEventBranches.branch(tree); }
  void initLuminosityBlockTree(TTree& tree) override { m_commonLumiBranches.branch(tree); }
  void initRunTree(TTree& tree) override { m_commonRunBranches.branch(tree); }

  class CommonEventBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
      tree.Branch("bunchCrossing", &m_bunchCrossing, "bunchCrossing/i");
      tree.Branch("orbitNumber", &m_orbitNumber, "orbitNumber/i");
    }
    void fill(edm::EventAuxiliary const& aux) {
      m_run = aux.id().run();
      m_luminosityBlock = aux.id().luminosityBlock();
      m_orbitNumber = aux.id().event();  // in L1Scouting, one processing event is one orbit
    }
    void setBx(unsigned int const bx) { m_bunchCrossing = bx; }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    UInt_t m_bunchCrossing;
    UInt_t m_orbitNumber;
  } m_commonEventBranches;

  class CommonLumiBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
      tree.Branch("nOrbits", &m_orbits, "nOrbits/i");
    }
    void fill(edm::LuminosityBlockID const& id, unsigned int const nOrbits) {
      m_run = id.run();
      m_luminosityBlock = id.value();
      m_orbits = nOrbits;
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    UInt_t m_orbits;
  } m_commonLumiBranches;

  class CommonRunBranches {
  public:
    void branch(TTree& tree) { tree.Branch("run", &m_run, "run/i"); }
    void fill(edm::RunID const& id) { m_run = id.run(); }

  private:
    UInt_t m_run;
  } m_commonRunBranches;

  bool const m_skipEmptyBXs;

  std::vector<OrbitTableOutputBranches> m_tables;
  std::vector<SelectedBxTableOutputBranches> m_selbxs;
  unsigned int m_nOrbits;

  edm::EDGetTokenT<std::vector<unsigned int>> m_bxMaskToken;
  std::vector<unsigned int> allBXs_;
};

OrbitNanoAODOutputModule::OrbitNanoAODOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      NanoAODOutputModuleBase(pset),
      m_skipEmptyBXs(pset.getParameter<bool>("skipEmptyBXs")),
      m_nOrbits(0) {
  auto const& bxMask = pset.getParameter<edm::InputTag>("selectedBx");
  if (!bxMask.label().empty()) {
    m_bxMaskToken = consumes(bxMask);
  } else {
    allBXs_.resize(l1ScoutingRun3::OrbitFlatTable::NBX);
    for (auto i = 0u; i < l1ScoutingRun3::OrbitFlatTable::NBX; ++i) {
      allBXs_[i] = i + 1;
    }
  }
}

void OrbitNanoAODOutputModule::writeEventTree(edm::EventForOutput const& iEvent) {
  ++m_nOrbits;

  m_commonEventBranches.fill(iEvent.eventAuxiliary());

  // fill all tables, starting from main tables and then doing extension tables
  for (auto extensions = 0u; extensions <= 1; ++extensions) {
    for (auto& t : m_tables) {
      t.beginFill(iEvent, *m_eventTree, extensions);
    }
  }

  // get m_table, read parameters, book branches, etc.
  for (auto& t : m_selbxs) {
    t.beginFill(iEvent, *m_eventTree);
  }

  // get vector of selected BXs to be filled
  auto const* selbx = &allBXs_;
  if (not m_bxMaskToken.isUninitialized()) {
    edm::Handle<std::vector<unsigned>> handle;
    iEvent.getByToken(m_bxMaskToken, handle);
    selbx = &*handle;
  }

  // convert from orbit as event to collision as event
  tbb::this_task_arena::isolate([&] {
    for (auto const bx : *selbx) {
      if (m_skipEmptyBXs) {
        bool empty = true;
        for (auto& t : m_tables) {
          if (t.hasBx(bx)) {
            empty = false;
            break;
          }
        }
        if (empty) {
          continue;
        }
      }

      m_commonEventBranches.setBx(bx);
      for (auto& t : m_tables) {
        t.fillBx(bx);
      }
      for (auto& t : m_selbxs) {
        t.fillBx(bx);
      }
      m_eventTree->Fill();
    }  // bx loop
  });

  // set entries of m_tables and m_selbxs to nullptr
  for (auto& t : m_tables) {
    t.endFill();
  }
  for (auto& t : m_selbxs) {
    t.endFill();
  }
}

void OrbitNanoAODOutputModule::writeLuminosityBlockTree(edm::LuminosityBlockForOutput const& iLumi) {
  m_commonLumiBranches.fill(iLumi.id(), m_nOrbits);
  m_nOrbits = 0;

  tbb::this_task_arena::isolate([&] { m_lumiTree->Fill(); });
}

void OrbitNanoAODOutputModule::writeRunTree(edm::RunForOutput const& iRun) {
  m_commonRunBranches.fill(iRun.id());

  tbb::this_task_arena::isolate([&] { m_runTree->Fill(); });
}

void OrbitNanoAODOutputModule::initTables() {
  m_tables.clear();
  auto const& keeps = keptProducts();
  for (auto const& keep : keeps[edm::InEvent]) {
    if (keep.first->className() == "l1ScoutingRun3::OrbitFlatTable")
      m_tables.emplace_back(keep.first, keep.second);
    else if (keep.first->className() == "std::vector<unsigned int>")
      m_selbxs.emplace_back(keep.first, keep.second);
    else
      throw cms::Exception("Configuration", "OrbitNanoAODOutputModule cannot handle class " + keep.first->className());
  }
}

void OrbitNanoAODOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  NanoAODOutputModuleBase::fillDescription(desc);

  edm::one::OutputModule<>::fillDescription(desc, {"drop *", "keep l1ScoutingRun3OrbitFlatTable_*Table_*_*"});

  desc.add<bool>("skipEmptyBXs", false)->setComment("Skip BXs where all input collections are empty");
  desc.add<edm::InputTag>("selectedBx", edm::InputTag())->setComment("selected Bx (1-3564)");

  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OrbitNanoAODOutputModule);
