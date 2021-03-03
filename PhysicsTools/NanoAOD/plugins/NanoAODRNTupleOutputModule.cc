// -*- C++ -*-
//
// Package:     PhysicsTools/NanoAODOutput
// Class  :     NanoAODRNTupleOutputModule
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Max Orok
//         Created:  Wed, 13 Jan 2021 14:21:41 GMT
//

#include <cstdint>
#include <string>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;

#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "PhysicsTools/NanoAOD/plugins/TableOutputFields.h"
#include "PhysicsTools/NanoAOD/plugins/TriggerOutputFields.h"

class NanoAODRNTupleOutputModule : public edm::one::OutputModule<> {
public:
  NanoAODRNTupleOutputModule(edm::ParameterSet const& pset);
  ~NanoAODRNTupleOutputModule() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void openFile(edm::FileBlock const&) override;
  bool isFileOpen() const override;
  void write(edm::EventForOutput const& e) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void writeRun(edm::RunForOutput const&) override;
  void reallyCloseFile() override;

  void initializeNTuple(edm::EventForOutput const& e);

  std::string m_fileName;
  std::string m_logicalFileName;
  edm::JobReport::Token m_jrToken;

  std::unique_ptr<RNTupleWriter> m_ntuple;
  TableCollections m_tables;
  std::vector<TriggerOutputFields> m_triggers;

  class CommonEventFields {
  public:
    void createFields(RNTupleModel& model) {
      model.AddField<UInt_t>("run", &m_run);
      model.AddField<UInt_t>("luminosityBlock", &m_luminosityBlock);
      model.AddField<std::uint64_t>("event", &m_event);
    }
    void fill(const edm::EventID& id) {
      m_run = id.run();
      m_luminosityBlock = id.luminosityBlock();
      m_event = id.event();
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    std::uint64_t m_event;
  } m_commonFields;
};

NanoAODRNTupleOutputModule::NanoAODRNTupleOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset),
      m_fileName(pset.getUntrackedParameter<std::string>("fileName")),
      m_logicalFileName(pset.getUntrackedParameter<std::string>("logicalFileName")) {}

NanoAODRNTupleOutputModule::~NanoAODRNTupleOutputModule() {}

void NanoAODRNTupleOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {}
void NanoAODRNTupleOutputModule::writeRun(edm::RunForOutput const& iRun) {}

bool NanoAODRNTupleOutputModule::isFileOpen() const {
  return nullptr != m_ntuple.get();
}

void NanoAODRNTupleOutputModule::openFile(edm::FileBlock const&) {
  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   // TODO check if needed
                                   //m_fakeName ? "PoolOutputModule" : "NanoAODOutputModule",
                                   "NanoAODRNTupleOutputModule",
                                   description().moduleLabel(),
                                   edm::createGlobalIdentifier(),
                                   std::string(),
                                   branchHash.digest().toString(),
                                   std::vector<std::string>());
}

void NanoAODRNTupleOutputModule::initializeNTuple(edm::EventForOutput const& iEvent) {
  // set up RNTuple schema
  auto model = RNTupleModel::Create();
  m_commonFields.createFields(*model);

  //m_tables.clear();
  const auto& keeps = keptProducts();
  for (const auto& keep: keeps[edm::InEvent]) {
    //std::cout << "branch name: " << keep.first->branchName() << "\n";
    if (keep.first->className() == "nanoaod::FlatTable") {
      edm::Handle<nanoaod::FlatTable> handle;
      const auto& token = keep.second;
      iEvent.getByToken(token, handle);
      m_tables.add(token, *handle);
    } else if (keep.first->className() == "edm::TriggerResults") {
      m_triggers.emplace_back(TriggerOutputFields(keep.first->processName(), keep.second));
    } else if (keep.first->className() == "std::basic_string<char,std::char_traits<char> >" &&
               keep.first->productInstanceName() == "genModel") {
      // m_evstrings.emplace_back(keep.first, keep.second, true);
    } else {
      throw cms::Exception("Configuration", "NanoAODOutputModule cannot handle class " + keep.first->className());
    }
  }
  m_tables.createFields(iEvent, *model);
  for (auto& trigger: m_triggers) {
    trigger.createFields(iEvent, *model);
  }
  m_tables.print();
  m_ntuple = RNTupleWriter::Recreate(std::move(model), "Events", m_fileName);
}

void NanoAODRNTupleOutputModule::write(edm::EventForOutput const& iEvent) {
  if (!m_ntuple) {
    initializeNTuple(iEvent);
  }

  edm::Service<edm::JobReport> jr;
  jr->eventWrittenToFile(m_jrToken, iEvent.id().run(), iEvent.id().event());

  m_commonFields.fill(iEvent.id());
  m_tables.fill(iEvent);
  for (auto& trigger: m_triggers) {
    trigger.fill(iEvent);
  }
  m_ntuple->Fill();
}

void NanoAODRNTupleOutputModule::reallyCloseFile() {
  // write ntuple to disk by calling the RNTupleWriter destructor
  m_ntuple.reset();

  edm::Service<edm::JobReport> jr;
  jr->outputFileClosed(m_jrToken);
}

void NanoAODRNTupleOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");

  const std::vector<std::string> keep = {"drop *",
                                         "keep nanoaodFlatTable_*Table_*_*",
                                         "keep edmTriggerResults_*_*_*",
                                         "keep String_*_genModel_*",
                                         "keep nanoaodMergeableCounterTable_*Table_*_*",
                                         "keep nanoaodUniqueString_nanoMetadata_*_*"};
  edm::one::OutputModule<>::fillDescription(desc, keep);

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(NanoAODRNTupleOutputModule);
