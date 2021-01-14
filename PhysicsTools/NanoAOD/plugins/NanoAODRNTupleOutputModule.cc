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

#include <string>

#include <ROOT/RNTupleModel.hxx>

#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class NanoAODRNTupleOutputModule : public edm::one::OutputModule<> {
public:
  NanoAODRNTupleOutputModule(edm::ParameterSet const& pset);
  ~NanoAODRNTupleOutputModule() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void write(edm::EventForOutput const& e) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void writeRun(edm::RunForOutput const&) override;
  bool isFileOpen() const override;
  void openFile(edm::FileBlock const&) override;
  void reallyCloseFile() override;
};

NanoAODRNTupleOutputModule::NanoAODRNTupleOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset) {}

NanoAODRNTupleOutputModule::~NanoAODRNTupleOutputModule() {}

void NanoAODRNTupleOutputModule::write(edm::EventForOutput const& iEvent) {}
void NanoAODRNTupleOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {}
void NanoAODRNTupleOutputModule::writeRun(edm::RunForOutput const& iRun) {}
bool NanoAODRNTupleOutputModule::isFileOpen() const { return false; }
void NanoAODRNTupleOutputModule::openFile(edm::FileBlock const&) {}
void NanoAODRNTupleOutputModule::reallyCloseFile() {}

void NanoAODRNTupleOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(NanoAODRNTupleOutputModule);
