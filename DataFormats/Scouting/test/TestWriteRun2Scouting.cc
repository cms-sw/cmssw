// -*- C++ -*-
//
// Package:    DataFormats/Scouting
// Class:      TestWriteRun2Scouting
//
/**\class edmtest::TestWriteRun2Scouting
  Description: Used as part of tests that ensure the Run 2 scouting
  data formats can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time a Run 2 Scouting persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  2 June 2023

#include "DataFormats/Scouting/interface/ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/ScoutingElectron.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingParticle.h"
#include "DataFormats/Scouting/interface/ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/ScoutingTrack.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteRun2Scouting : public edm::global::EDProducer<> {
  public:
    TestWriteRun2Scouting(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produceCaloJets(edm::Event&) const;
    void produceElectrons(edm::Event&) const;
    void produceMuons(edm::Event&) const;
    void produceParticles(edm::Event&) const;
    void producePFJets(edm::Event&) const;
    void producePhotons(edm::Event&) const;
    void produceTracks(edm::Event&) const;
    void produceVertexes(edm::Event&) const;

    void throwWithMessage(const char*) const;

    const std::vector<double> caloJetsValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingCaloJet>> caloJetsPutToken_;

    const std::vector<double> electronsFloatingPointValues_;
    const std::vector<int> electronsIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingElectron>> electronsPutToken_;

    const std::vector<double> muonsFloatingPointValues_;
    const std::vector<int> muonsIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingMuon>> muonsPutToken_;

    const std::vector<double> particlesFloatingPointValues_;
    const std::vector<int> particlesIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingParticle>> particlesPutToken_;

    const std::vector<double> pfJetsFloatingPointValues_;
    const std::vector<int> pfJetsIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingPFJet>> pfJetsPutToken_;

    const std::vector<double> photonsFloatingPointValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingPhoton>> photonsPutToken_;

    const std::vector<double> tracksFloatingPointValues_;
    const std::vector<int> tracksIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingTrack>> tracksPutToken_;

    const std::vector<double> vertexesFloatingPointValues_;
    const std::vector<int> vertexesIntegralValues_;
    //const edm::EDPutTokenT<std::vector<ScoutingVertex>> vertexesPutToken_;
  };

  TestWriteRun2Scouting::TestWriteRun2Scouting(edm::ParameterSet const& iPSet)
      : caloJetsValues_(iPSet.getParameter<std::vector<double>>("caloJetsValues")),
        //caloJetsPutToken_(produces()),
        electronsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("electronsFloatingPointValues")),
        electronsIntegralValues_(iPSet.getParameter<std::vector<int>>("electronsIntegralValues")),
        //electronsPutToken_(produces()),
        muonsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("muonsFloatingPointValues")),
        muonsIntegralValues_(iPSet.getParameter<std::vector<int>>("muonsIntegralValues")),
        //muonsPutToken_(produces()),
        particlesFloatingPointValues_(iPSet.getParameter<std::vector<double>>("particlesFloatingPointValues")),
        particlesIntegralValues_(iPSet.getParameter<std::vector<int>>("particlesIntegralValues")),
        //particlesPutToken_(produces()),
        pfJetsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("pfJetsFloatingPointValues")),
        pfJetsIntegralValues_(iPSet.getParameter<std::vector<int>>("pfJetsIntegralValues")),
        //pfJetsPutToken_(produces()),
        photonsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("photonsFloatingPointValues")),
        //photonsPutToken_(produces()),
        tracksFloatingPointValues_(iPSet.getParameter<std::vector<double>>("tracksFloatingPointValues")),
        tracksIntegralValues_(iPSet.getParameter<std::vector<int>>("tracksIntegralValues")),
        //tracksPutToken_(produces()),
        vertexesFloatingPointValues_(iPSet.getParameter<std::vector<double>>("vertexesFloatingPointValues")),
        vertexesIntegralValues_(iPSet.getParameter<std::vector<int>>("vertexesIntegralValues")) {
        //vertexesIntegralValues_(iPSet.getParameter<std::vector<int>>("vertexesIntegralValues")),
        //vertexesPutToken_(produces()) {
    produces<std::vector<ScoutingCaloJet>>();
    produces<std::vector<ScoutingElectron>>();
    produces<std::vector<ScoutingMuon>>();
    produces<std::vector<ScoutingParticle>>();
    produces<std::vector<ScoutingPFJet>>();
    produces<std::vector<ScoutingPhoton>>();
    produces<std::vector<ScoutingTrack>>();
    produces<std::vector<ScoutingVertex>>();
    if (caloJetsValues_.size() != 16) {
      throwWithMessage("caloJetsValues must have 16 elements and it does not");
    }
    if (electronsFloatingPointValues_.size() != 14) {
      throwWithMessage("electronsFloatingPointValues must have 14 elements and it does not");
    }
    if (electronsIntegralValues_.size() != 2) {
      throwWithMessage("electronsIntegralValues must have 2 elements and it does not");
    }
    if (muonsFloatingPointValues_.size() != 23) {
      throwWithMessage("muonsFloatingPointValues must have 23 elements and it does not");
    }
    if (muonsIntegralValues_.size() != 8) {
      throwWithMessage("muonsIntegralValues must have 8 elements and it does not");
    }
    if (particlesFloatingPointValues_.size() != 4) {
      throwWithMessage("particlesFloatingPointValues must have 4 elements and it does not");
    }
    if (particlesIntegralValues_.size() != 2) {
      throwWithMessage("particlesIntegralValues must have 2 elements and it does not");
    }
    if (pfJetsFloatingPointValues_.size() != 15) {
      throwWithMessage("pfJetsFloatingPointValues must have 15 elements and it does not");
    }
    if (pfJetsIntegralValues_.size() != 8) {
      throwWithMessage("pfJetsIntegralValues must have 8 elements and it does not");
    }
    if (photonsFloatingPointValues_.size() != 8) {
      throwWithMessage("photonsFloatingPointValues must have 8 elements and it does not");
    }
    if (tracksFloatingPointValues_.size() != 16) {
      throwWithMessage("tracksFloatingPointValues must have 16 elements and it does not");
    }
    if (tracksIntegralValues_.size() != 4) {
      throwWithMessage("tracksIntegralValues must have 4 elements and it does not");
    }
    if (vertexesFloatingPointValues_.size() != 7) {
      throwWithMessage("vertexesFloatingPointValues must have 7 elements and it does not");
    }
    if (vertexesIntegralValues_.size() != 3) {
      throwWithMessage("vertexesIntegralValues must have 3 elements and it does not");
    }
  }

  void TestWriteRun2Scouting::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill Run2 scouting objects. Make sure all the containers inside
    // of them have something in them (not empty). The values are meaningless.
    // We will later check that after writing these objects to persistent storage
    // and then reading them in a later process we obtain matching values for
    // all this content.

    produceCaloJets(iEvent);
    produceElectrons(iEvent);
    produceMuons(iEvent);
    produceParticles(iEvent);
    producePFJets(iEvent);
    producePhotons(iEvent);
    produceTracks(iEvent);
    produceVertexes(iEvent);
  }

  void TestWriteRun2Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<double>>("caloJetsValues");
    desc.add<std::vector<double>>("electronsFloatingPointValues");
    desc.add<std::vector<int>>("electronsIntegralValues");
    desc.add<std::vector<double>>("muonsFloatingPointValues");
    desc.add<std::vector<int>>("muonsIntegralValues");
    desc.add<std::vector<double>>("particlesFloatingPointValues");
    desc.add<std::vector<int>>("particlesIntegralValues");
    desc.add<std::vector<double>>("pfJetsFloatingPointValues");
    desc.add<std::vector<int>>("pfJetsIntegralValues");
    desc.add<std::vector<double>>("photonsFloatingPointValues");
    desc.add<std::vector<double>>("tracksFloatingPointValues");
    desc.add<std::vector<int>>("tracksIntegralValues");
    desc.add<std::vector<double>>("vertexesFloatingPointValues");
    desc.add<std::vector<int>>("vertexesIntegralValues");
    descriptions.addDefault(desc);
  }

  void TestWriteRun2Scouting::produceCaloJets(edm::Event& iEvent) const {
    auto run2ScoutingCaloJets = std::make_unique<std::vector<ScoutingCaloJet>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingCaloJets->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      run2ScoutingCaloJets->emplace_back(static_cast<float>(caloJetsValues_[0] + offset),
                                         static_cast<float>(caloJetsValues_[1] + offset),
                                         static_cast<float>(caloJetsValues_[2] + offset),
                                         static_cast<float>(caloJetsValues_[3] + offset),
                                         static_cast<float>(caloJetsValues_[4] + offset),
                                         static_cast<float>(caloJetsValues_[5] + offset),
                                         static_cast<float>(caloJetsValues_[6] + offset),
                                         static_cast<float>(caloJetsValues_[7] + offset),
                                         static_cast<float>(caloJetsValues_[8] + offset),
                                         static_cast<float>(caloJetsValues_[9] + offset),
                                         static_cast<float>(caloJetsValues_[10] + offset),
                                         static_cast<float>(caloJetsValues_[11] + offset),
                                         static_cast<float>(caloJetsValues_[12] + offset),
                                         static_cast<float>(caloJetsValues_[13] + offset),
                                         static_cast<float>(caloJetsValues_[14] + offset),
                                         static_cast<float>(caloJetsValues_[15] + offset));
    }
    //iEvent.put(caloJetsPutToken_, std::move(run2ScoutingCaloJets));
    iEvent.put(std::move(run2ScoutingCaloJets));
  }

  void TestWriteRun2Scouting::produceElectrons(edm::Event& iEvent) const {
    auto run2ScoutingElectrons = std::make_unique<std::vector<ScoutingElectron>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingElectrons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      run2ScoutingElectrons->emplace_back(static_cast<float>(electronsFloatingPointValues_[0] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[1] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[2] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[3] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[4] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[5] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[6] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[7] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[8] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[9] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[10] + offset),
                                          electronsIntegralValues_[0] + iOffset,
                                          electronsIntegralValues_[1] + iOffset,
                                          static_cast<float>(electronsFloatingPointValues_[11] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[12] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[13] + offset));
    }
    //iEvent.put(electronsPutToken_, std::move(run2ScoutingElectrons));
    iEvent.put(std::move(run2ScoutingElectrons));
  }

  void TestWriteRun2Scouting::produceMuons(edm::Event& iEvent) const {
    auto run2ScoutingMuons = std::make_unique<std::vector<ScoutingMuon>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingMuons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      std::vector<int> vtxIndx;
      vtxIndx.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        vtxIndx.push_back(static_cast<int>(muonsIntegralValues_[7] + iOffset + j * 10));
      }

      run2ScoutingMuons->emplace_back(static_cast<float>(muonsFloatingPointValues_[0] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[1] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[2] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[3] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[4] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[5] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[6] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[7] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[8] + offset),
                                      muonsIntegralValues_[0] + iOffset,
                                      static_cast<float>(muonsFloatingPointValues_[9] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[10] + offset),
                                      muonsIntegralValues_[1] + iOffset,
                                      muonsIntegralValues_[2] + iOffset,
                                      muonsIntegralValues_[3] + iOffset,
                                      muonsIntegralValues_[4] + iOffset,
                                      muonsIntegralValues_[5] + iOffset,
                                      muonsIntegralValues_[6] + iOffset,
                                      static_cast<float>(muonsFloatingPointValues_[11] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[12] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[13] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[14] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[15] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[16] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[17] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[18] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[19] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[20] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[21] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[22] + offset),
                                      std::move(vtxIndx));
    }
    //iEvent.put(muonsPutToken_, std::move(run2ScoutingMuons));
    iEvent.put(std::move(run2ScoutingMuons));
  }

  void TestWriteRun2Scouting::produceParticles(edm::Event& iEvent) const {
    auto run2ScoutingParticles = std::make_unique<std::vector<ScoutingParticle>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingParticles->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);
      run2ScoutingParticles->emplace_back(static_cast<float>(particlesFloatingPointValues_[0] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[1] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[2] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[3] + offset),
                                          particlesIntegralValues_[0] + iOffset,
                                          particlesIntegralValues_[1] + iOffset);
    }
    //iEvent.put(particlesPutToken_, std::move(run2ScoutingParticles));
    iEvent.put(std::move(run2ScoutingParticles));
  }

  void TestWriteRun2Scouting::producePFJets(edm::Event& iEvent) const {
    auto run2ScoutingPFJets = std::make_unique<std::vector<ScoutingPFJet>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingPFJets->reserve(vectorSize);

    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      std::vector<int> constituents;
      constituents.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        constituents.push_back(static_cast<int>(pfJetsIntegralValues_[7] + iOffset + j * 10));
      }

      run2ScoutingPFJets->emplace_back(static_cast<float>(pfJetsFloatingPointValues_[0] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[1] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[2] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[3] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[4] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[5] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[6] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[7] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[8] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[9] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[10] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[11] + offset),
                                       pfJetsIntegralValues_[0] + iOffset,
                                       pfJetsIntegralValues_[1] + iOffset,
                                       pfJetsIntegralValues_[2] + iOffset,
                                       pfJetsIntegralValues_[3] + iOffset,
                                       pfJetsIntegralValues_[4] + iOffset,
                                       pfJetsIntegralValues_[5] + iOffset,
                                       pfJetsIntegralValues_[6] + iOffset,
                                       static_cast<float>(pfJetsFloatingPointValues_[12] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[13] + offset),
                                       static_cast<float>(pfJetsFloatingPointValues_[14] + offset),
                                       std::move(constituents));
    }
    //iEvent.put(pfJetsPutToken_, std::move(run2ScoutingPFJets));
    iEvent.put(std::move(run2ScoutingPFJets));
  }

  void TestWriteRun2Scouting::producePhotons(edm::Event& iEvent) const {
    auto run2ScoutingPhotons = std::make_unique<std::vector<ScoutingPhoton>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingPhotons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);

      run2ScoutingPhotons->emplace_back(static_cast<float>(photonsFloatingPointValues_[0] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[1] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[2] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[3] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[4] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[5] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[6] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[7] + offset));
    }
    //iEvent.put(photonsPutToken_, std::move(run2ScoutingPhotons));
    iEvent.put(std::move(run2ScoutingPhotons));
  }

  void TestWriteRun2Scouting::produceTracks(edm::Event& iEvent) const {
    auto run2ScoutingTracks = std::make_unique<std::vector<ScoutingTrack>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingTracks->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      run2ScoutingTracks->emplace_back(static_cast<float>(tracksFloatingPointValues_[0] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[1] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[2] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[3] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[4] + offset),
                                       tracksIntegralValues_[0] + iOffset,
                                       static_cast<float>(tracksFloatingPointValues_[5] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[6] + offset),
                                       tracksIntegralValues_[1] + iOffset,
                                       tracksIntegralValues_[2] + iOffset,
                                       tracksIntegralValues_[3] + iOffset,
                                       static_cast<float>(tracksFloatingPointValues_[7] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[8] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[9] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[10] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[11] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[12] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[13] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[14] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[15] + offset));
    }
    //iEvent.put(tracksPutToken_, std::move(run2ScoutingTracks));
    iEvent.put(std::move(run2ScoutingTracks));
  }

  void TestWriteRun2Scouting::produceVertexes(edm::Event& iEvent) const {
    auto run2ScoutingVertexes = std::make_unique<std::vector<ScoutingVertex>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run2ScoutingVertexes->reserve(vectorSize);

    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      run2ScoutingVertexes->emplace_back(static_cast<float>(vertexesFloatingPointValues_[0] + offset),
                                         static_cast<float>(vertexesFloatingPointValues_[1] + offset),
                                         static_cast<float>(vertexesFloatingPointValues_[2] + offset),
                                         static_cast<float>(vertexesFloatingPointValues_[3] + offset),
                                         static_cast<float>(vertexesFloatingPointValues_[4] + offset),
                                         static_cast<float>(vertexesFloatingPointValues_[5] + offset),
                                         vertexesIntegralValues_[0] + iOffset,
                                         static_cast<float>(vertexesFloatingPointValues_[6] + offset),
                                         vertexesIntegralValues_[1] + iOffset,
                                         static_cast<bool>((vertexesIntegralValues_[2] + iOffset) % 2));
    }
    //iEvent.put(vertexesPutToken_, std::move(run2ScoutingVertexes));
    iEvent.put(std::move(run2ScoutingVertexes));
  }

  void TestWriteRun2Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestWriteRun2Scouting constructor, test configuration error, " << msg;
  }

}  // namespace edmtest

using edmtest::TestWriteRun2Scouting;
DEFINE_FWK_MODULE(TestWriteRun2Scouting);
