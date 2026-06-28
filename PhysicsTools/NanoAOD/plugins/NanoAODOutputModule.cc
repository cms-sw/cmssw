// -*- C++ -*-
//
// Package:     PhysicsTools/NanoAODOutput
// Class  :     NanoAODOutputModule
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 07 Aug 2017 14:21:41 GMT
//
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "TObjString.h"

#include "oneapi/tbb/task_arena.h"

#include "DataFormats/NanoAOD/interface/UniqueString.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/NanoAOD/interface/NanoAODOutputModuleBase.h"

#include "EventStringOutputBranches.h"
#include "LumiOutputBranches.h"
#include "TableOutputBranches.h"
#include "TriggerOutputBranches.h"
#include "SummaryTableOutputBranches.h"

class NanoAODOutputModule : public NanoAODOutputModuleBase {
public:
  explicit NanoAODOutputModule(edm::ParameterSet const& pset);

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
      tree.Branch("event", &m_event, "event/l");
      tree.Branch("bunchCrossing", &m_bunchCrossing, "bunchCrossing/i");
      tree.Branch("orbitNumber", &m_orbitNumber, "orbitNumber/i");
    }
    void fill(edm::EventAuxiliary const& aux) {
      m_run = aux.id().run();
      m_luminosityBlock = aux.id().luminosityBlock();
      m_event = aux.id().event();
      m_bunchCrossing = aux.bunchCrossing();
      m_orbitNumber = aux.orbitNumber();
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    ULong64_t m_event;
    UInt_t m_bunchCrossing;
    UInt_t m_orbitNumber;
  } m_commonEventBranches;

  class CommonLumiBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
    }
    void fill(edm::LuminosityBlockID const& id) {
      m_run = id.run();
      m_luminosityBlock = id.value();
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
  } m_commonLumiBranches;

  class CommonRunBranches {
  public:
    void branch(TTree& tree) { tree.Branch("run", &m_run, "run/i"); }
    void fill(edm::RunID const& id) { m_run = id.run(); }

  private:
    UInt_t m_run;
  } m_commonRunBranches;

  std::vector<TableOutputBranches> m_tables;
  std::vector<TriggerOutputBranches> m_triggers;
  bool m_triggers_areSorted = false;
  std::vector<EventStringOutputBranches> m_evstrings;

  std::vector<SummaryTableOutputBranches> m_runTables;
  std::vector<SummaryTableOutputBranches> m_lumiTables;
  std::vector<LumiOutputBranches> m_lumiTables2;
  std::vector<TableOutputBranches> m_runFlatTables;

  std::vector<std::pair<std::string, edm::EDGetToken>> m_nanoMetadata;
};

NanoAODOutputModule::NanoAODOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset), NanoAODOutputModuleBase(pset) {}

void NanoAODOutputModule::writeEventTree(edm::EventForOutput const& iEvent) {
  m_commonEventBranches.fill(iEvent.eventAuxiliary());
  // fill all tables, starting from main tables and then doing extension tables
  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_tables)
      t.fill(iEvent, *m_eventTree, extensions);
  }
  if (!m_triggers_areSorted) {  // sort triggers/flags in inverse processHistory order, to save without any special label the most recent ones
    std::vector<std::string> pnames;
    for (auto const& p : iEvent.processHistory())
      pnames.push_back(p.processName());
    std::sort(m_triggers.begin(), m_triggers.end(), [pnames](TriggerOutputBranches& a, TriggerOutputBranches& b) {
      return ((std::find(pnames.begin(), pnames.end(), a.processName()) - pnames.begin()) >
              (std::find(pnames.begin(), pnames.end(), b.processName()) - pnames.begin()));
    });
    m_triggers_areSorted = true;
  }
  // fill triggers
  for (auto& t : m_triggers)
    t.fill(iEvent, *m_eventTree);
  // fill event branches
  for (auto& t : m_evstrings)
    t.fill(iEvent, *m_eventTree);

  tbb::this_task_arena::isolate([&] { m_eventTree->Fill(); });
}

void NanoAODOutputModule::writeLuminosityBlockTree(edm::LuminosityBlockForOutput const& iLumi) {
  m_commonLumiBranches.fill(iLumi.id());

  for (auto& t : m_lumiTables) {
    t.fill(iLumi, *m_lumiTree);
  }

  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_lumiTables2) {
      t.fill(iLumi, *m_lumiTree, extensions);
    }
  }

  tbb::this_task_arena::isolate([&] { m_lumiTree->Fill(); });
}

void NanoAODOutputModule::writeRunTree(edm::RunForOutput const& iRun) {
  m_commonRunBranches.fill(iRun.id());

  for (auto& t : m_runTables) {
    t.fill(iRun, *m_runTree);
  }

  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_runFlatTables) {
      t.fill(iRun, *m_runTree, extensions);
    }
  }

  edm::Handle<nanoaod::UniqueString> hstring;
  for (auto const& p : m_nanoMetadata) {
    iRun.getByToken(p.second, hstring);
    TObjString* tos = dynamic_cast<TObjString*>(m_file->Get(p.first.c_str()));
    if (tos) {
      if (hstring->str() != tos->GetString())
        throw cms::Exception("LogicError", "Inconsistent nanoMetadata " + p.first + " (" + hstring->str() + ")");
    } else {
      auto ostr = std::make_unique<TObjString>(hstring->str().c_str());
      m_file->WriteTObject(ostr.get(), p.first.c_str());
    }
  }

  tbb::this_task_arena::isolate([&] { m_runTree->Fill(); });
}

void NanoAODOutputModule::initTables() {
  m_tables.clear();
  m_triggers.clear();
  m_triggers_areSorted = false;
  m_evstrings.clear();
  m_runTables.clear();
  m_lumiTables.clear();
  m_lumiTables2.clear();
  m_runFlatTables.clear();
  auto const& keeps = keptProducts();
  for (auto const& keep : keeps[edm::InEvent]) {
    if (keep.first->className() == "nanoaod::FlatTable")
      m_tables.emplace_back(keep.first, keep.second);
    else if (keep.first->className() == "edm::TriggerResults") {
      m_triggers.emplace_back(keep.first, keep.second);
    } else if (keep.first->className() == "std::basic_string<char,std::char_traits<char> >" &&
               keep.first->productInstanceName() == "genModel") {  // friendlyClassName == "String"
      m_evstrings.emplace_back(keep.first, keep.second, true);     // update only at lumiBlock transitions
    } else
      throw cms::Exception("Configuration", "NanoAODOutputModule cannot handle class " + keep.first->className());
  }

  for (auto const& keep : keeps[edm::InLumi]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable")
      m_lumiTables.push_back(SummaryTableOutputBranches(keep.first, keep.second));
    else if (keep.first->className() == "nanoaod::UniqueString" && keep.first->moduleLabel() == "nanoMetadata")
      m_nanoMetadata.emplace_back(keep.first->productInstanceName(), keep.second);
    else if (keep.first->className() == "nanoaod::FlatTable")
      m_lumiTables2.push_back(LumiOutputBranches(keep.first, keep.second));
    else
      throw cms::Exception(
          "Configuration",
          "NanoAODOutputModule cannot handle class " + keep.first->className() + " in LuminosityBlock branch");
  }

  for (auto const& keep : keeps[edm::InRun]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable")
      m_runTables.push_back(SummaryTableOutputBranches(keep.first, keep.second));
    else if (keep.first->className() == "nanoaod::UniqueString" && keep.first->moduleLabel() == "nanoMetadata")
      m_nanoMetadata.emplace_back(keep.first->productInstanceName(), keep.second);
    else if (keep.first->className() == "nanoaod::FlatTable")
      m_runFlatTables.emplace_back(keep.first, keep.second);
    else
      throw cms::Exception("Configuration",
                           "NanoAODOutputModule cannot handle class " + keep.first->className() + " in Run branch");
  }
}

void NanoAODOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  NanoAODOutputModuleBase::fillDescription(desc);

  edm::one::OutputModule<>::fillDescription(desc,
                                            {"drop *",
                                             "keep nanoaodFlatTable_*Table_*_*",
                                             "keep edmTriggerResults_*_*_*",
                                             "keep String_*_genModel_*",
                                             "keep nanoaodMergeableCounterTable_*Table_*_*",
                                             "keep nanoaodUniqueString_nanoMetadata_*_*"});

  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NanoAODOutputModule);
