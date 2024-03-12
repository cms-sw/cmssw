// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      LHCInfoProducer
//
/**\class LHCInfoProducer LHCInfoProducer.cc PhysicsTools/NanoAOD/plugins/LHCInfoProducer.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Justin Williams
//         Created: 05 Jul 2019 14:06:12 GMT
//
//

// System include files
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CommonTools/Egamma/interface/EffectiveAreas.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"

#include "FWCore/Utilities/interface/transform.h"

#include "CondTools/RunInfo/interface/LHCInfoCombined.h"

class LHCInfoProducer : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
public:
  LHCInfoProducer(edm::ParameterSet const& iConfig)
      : lhcinfoToken_(esConsumes<edm::Transition::BeginLuminosityBlock>(
            edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
        lhcinfoPerLSToken_(esConsumes<edm::Transition::BeginLuminosityBlock>(
            edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")))),
        lhcinfoPerFillToken_(esConsumes<edm::Transition::BeginLuminosityBlock>(
            edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")))),
        useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")) {
    produces<nanoaod::MergeableCounterTable, edm::Transition::BeginLuminosityBlock>();
  }
  ~LHCInfoProducer() override {}

  // ------------ method called to produce the data  ------------
  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {}

  void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& iLumi, edm::EventSetup const& iSetup) const override {
    LHCInfoCombined lhcInfoCombined(iSetup, lhcinfoPerLSToken_, lhcinfoPerFillToken_, lhcinfoToken_, useNewLHCInfo_);
    auto out = std::make_unique<nanoaod::MergeableCounterTable>();
    out->addFloat("crossingAngle", "LHC crossing angle", lhcInfoCombined.crossingAngle());
    out->addFloat("betaStar", "LHC beta star", lhcInfoCombined.betaStarX);
    out->addFloat("energy", "LHC beam energy", lhcInfoCombined.energy);
    iLumi.put(std::move(out));
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("lhcInfoLabel", "")->setComment("label used for LHCInfo");
    desc.add<std::string>("lhcInfoPerLSLabel", "")->setComment("label of the LHCInfoPerLS record");
    desc.add<std::string>("lhcInfoPerFillLabel", "")->setComment("label of the LHCInfoPerFill record");
    desc.add<bool>("useNewLHCInfo", true)
        ->setComment("flag whether to use new LHCInfoPerLS/Fill records or old LHCInfo");
    descriptions.addWithDefaultLabel(desc);
  }
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcinfoToken_;
  edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcinfoPerLSToken_;
  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcinfoPerFillToken_;
  bool useNewLHCInfo_;
};

DEFINE_FWK_MODULE(LHCInfoProducer);
