// -*- C++ -*-
//
// Package:    DataFormats/Scouting
// Class:      TestWriteRun3Scouting
//
/**\class edmtest::TestWriteRun3Scouting
  Description: Used as part of tests that ensure the Run 3 scouting
  data formats can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time a Run 3 Scouting persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  17 May 2023

#include "DataFormats/Scouting/interface/Run3ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingHitPatternPOD.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteRun3Scouting : public edm::global::EDProducer<> {
  public:
    TestWriteRun3Scouting(edm::ParameterSet const&);
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
    const edm::EDPutTokenT<std::vector<Run3ScoutingCaloJet>> caloJetsPutToken_;

    const std::vector<double> electronsFloatingPointValues_;
    const std::vector<int> electronsIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingElectron>> electronsPutToken_;

    const std::vector<double> muonsFloatingPointValues_;
    const std::vector<int> muonsIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingMuon>> muonsPutToken_;

    const std::vector<double> particlesFloatingPointValues_;
    const std::vector<int> particlesIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingParticle>> particlesPutToken_;

    const std::vector<double> pfJetsFloatingPointValues_;
    const std::vector<int> pfJetsIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingPFJet>> pfJetsPutToken_;

    const std::vector<double> photonsFloatingPointValues_;
    const std::vector<int> photonsIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingPhoton>> photonsPutToken_;

    const std::vector<double> tracksFloatingPointValues_;
    const std::vector<int> tracksIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingTrack>> tracksPutToken_;

    const std::vector<double> vertexesFloatingPointValues_;
    const std::vector<int> vertexesIntegralValues_;
    const edm::EDPutTokenT<std::vector<Run3ScoutingVertex>> vertexesPutToken_;
  };

  TestWriteRun3Scouting::TestWriteRun3Scouting(edm::ParameterSet const& iPSet)
      : caloJetsValues_(iPSet.getParameter<std::vector<double>>("caloJetsValues")),
        caloJetsPutToken_(produces()),
        electronsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("electronsFloatingPointValues")),
        electronsIntegralValues_(iPSet.getParameter<std::vector<int>>("electronsIntegralValues")),
        electronsPutToken_(produces()),
        muonsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("muonsFloatingPointValues")),
        muonsIntegralValues_(iPSet.getParameter<std::vector<int>>("muonsIntegralValues")),
        muonsPutToken_(produces()),
        particlesFloatingPointValues_(iPSet.getParameter<std::vector<double>>("particlesFloatingPointValues")),
        particlesIntegralValues_(iPSet.getParameter<std::vector<int>>("particlesIntegralValues")),
        particlesPutToken_(produces()),
        pfJetsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("pfJetsFloatingPointValues")),
        pfJetsIntegralValues_(iPSet.getParameter<std::vector<int>>("pfJetsIntegralValues")),
        pfJetsPutToken_(produces()),
        photonsFloatingPointValues_(iPSet.getParameter<std::vector<double>>("photonsFloatingPointValues")),
        photonsIntegralValues_(iPSet.getParameter<std::vector<int>>("photonsIntegralValues")),
        photonsPutToken_(produces()),
        tracksFloatingPointValues_(iPSet.getParameter<std::vector<double>>("tracksFloatingPointValues")),
        tracksIntegralValues_(iPSet.getParameter<std::vector<int>>("tracksIntegralValues")),
        tracksPutToken_(produces()),
        vertexesFloatingPointValues_(iPSet.getParameter<std::vector<double>>("vertexesFloatingPointValues")),
        vertexesIntegralValues_(iPSet.getParameter<std::vector<int>>("vertexesIntegralValues")),
        vertexesPutToken_(produces()) {
    if (caloJetsValues_.size() != 16) {
      throwWithMessage("caloJetsValues must have 16 elements and it does not");
    }
    if (electronsFloatingPointValues_.size() != 25) {
      throwWithMessage("electronsFloatingPointValues must have 25 elements and it does not");
    }
    if (electronsIntegralValues_.size() != 6) {
      throwWithMessage("electronsIntegralValues must have 6 elements and it does not");
    }
    if (muonsFloatingPointValues_.size() != 37) {
      throwWithMessage("muonsFloatingPointValues must have 37 elements and it does not");
    }
    if (muonsIntegralValues_.size() != 26) {
      throwWithMessage("muonsIntegralValues must have 26 elements and it does not");
    }
    if (particlesFloatingPointValues_.size() != 11) {
      throwWithMessage("particlesFloatingPointValues must have 11 elements and it does not");
    }
    if (particlesIntegralValues_.size() != 5) {
      throwWithMessage("particlesIntegralValues must have 5 elements and it does not");
    }
    if (pfJetsFloatingPointValues_.size() != 15) {
      throwWithMessage("pfJetsFloatingPointValues must have 15 elements and it does not");
    }
    if (pfJetsIntegralValues_.size() != 8) {
      throwWithMessage("pfJetsIntegralValues must have 8 elements and it does not");
    }
    if (photonsFloatingPointValues_.size() != 14) {
      throwWithMessage("photonsFloatingPointValues must have 14 elements and it does not");
    }
    if (photonsIntegralValues_.size() != 3) {
      throwWithMessage("photonsIntegralValues must have 3 elements and it does not");
    }
    if (tracksFloatingPointValues_.size() != 29) {
      throwWithMessage("tracksFloatingPointValues must have 29 elements and it does not");
    }
    if (tracksIntegralValues_.size() != 5) {
      throwWithMessage("tracksIntegralValues must have 5 elements and it does not");
    }
    if (vertexesFloatingPointValues_.size() != 7) {
      throwWithMessage("vertexesFloatingPointValues must have 7 elements and it does not");
    }
    if (vertexesIntegralValues_.size() != 3) {
      throwWithMessage("vertexesIntegralValues must have 3 elements and it does not");
    }
  }

  void TestWriteRun3Scouting::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill Run3 scouting objects. Make sure all the containers inside
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

  void TestWriteRun3Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
    desc.add<std::vector<int>>("photonsIntegralValues");
    desc.add<std::vector<double>>("tracksFloatingPointValues");
    desc.add<std::vector<int>>("tracksIntegralValues");
    desc.add<std::vector<double>>("vertexesFloatingPointValues");
    desc.add<std::vector<int>>("vertexesIntegralValues");
    descriptions.addDefault(desc);
  }

  void TestWriteRun3Scouting::produceCaloJets(edm::Event& iEvent) const {
    auto run3ScoutingCaloJets = std::make_unique<std::vector<Run3ScoutingCaloJet>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingCaloJets->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      run3ScoutingCaloJets->emplace_back(static_cast<float>(caloJetsValues_[0] + offset),
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
    iEvent.put(caloJetsPutToken_, std::move(run3ScoutingCaloJets));
  }

  void TestWriteRun3Scouting::produceElectrons(edm::Event& iEvent) const {
    auto run3ScoutingElectrons = std::make_unique<std::vector<Run3ScoutingElectron>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingElectrons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      // Note the first seven of these vectors use an out of sequence index
      // (starting at 19 or 5) because they are data members added in a format
      // change. In the CMSSW_12_4_0 version, they didn't exist.
      // Also the index values 4 and 5 in electronsFloatingPointValues_
      // and index 1 in electronsIntegralValues_ are not used because
      // those data members were dropped in the same format change.
      std::vector<float> trkd0;
      std::vector<float> trkdz;
      std::vector<float> trkpt;
      std::vector<float> trketa;
      std::vector<float> trkphi;
      std::vector<float> trkchi2overndf;
      std::vector<int> trkcharge;
      std::vector<float> energyMatrix;
      std::vector<unsigned int> detIds;
      std::vector<float> timingMatrix;
      trkd0.reserve(vectorSize);
      trkdz.reserve(vectorSize);
      trkpt.reserve(vectorSize);
      trketa.reserve(vectorSize);
      trkphi.reserve(vectorSize);
      trkchi2overndf.reserve(vectorSize);
      trkcharge.reserve(vectorSize);
      energyMatrix.reserve(vectorSize);
      detIds.reserve(vectorSize);
      timingMatrix.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        trkd0.push_back(static_cast<float>(electronsFloatingPointValues_[19] + offset + j * 10));
        trkdz.push_back(static_cast<float>(electronsFloatingPointValues_[20] + offset + j * 10));
        trkpt.push_back(static_cast<float>(electronsFloatingPointValues_[21] + offset + j * 10));
        trketa.push_back(static_cast<float>(electronsFloatingPointValues_[22] + offset + j * 10));
        trkphi.push_back(static_cast<float>(electronsFloatingPointValues_[23] + offset + j * 10));
        trkchi2overndf.push_back(static_cast<float>(electronsFloatingPointValues_[24] + offset + j * 10));
        trkcharge.push_back(static_cast<int>(electronsIntegralValues_[5] + offset + j * 10));
        energyMatrix.push_back(static_cast<float>(electronsFloatingPointValues_[17] + offset + j * 10));
        detIds.push_back(static_cast<uint32_t>(electronsIntegralValues_[3] + iOffset + j * 10));
        timingMatrix.push_back(static_cast<float>(electronsFloatingPointValues_[18] + offset + j * 10));
      }
      run3ScoutingElectrons->emplace_back(static_cast<float>(electronsFloatingPointValues_[0] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[1] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[2] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[3] + offset),
                                          std::move(trkd0),
                                          std::move(trkdz),
                                          std::move(trkpt),
                                          std::move(trketa),
                                          std::move(trkphi),
                                          std::move(trkchi2overndf),
                                          static_cast<float>(electronsFloatingPointValues_[6] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[7] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[8] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[9] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[10] + offset),
                                          electronsIntegralValues_[0] + iOffset,
                                          std::move(trkcharge),
                                          static_cast<float>(electronsFloatingPointValues_[11] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[12] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[13] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[14] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[15] + offset),
                                          static_cast<float>(electronsFloatingPointValues_[16] + offset),
                                          static_cast<uint32_t>(electronsIntegralValues_[2] + iOffset),
                                          std::move(energyMatrix),
                                          std::move(detIds),
                                          std::move(timingMatrix),
                                          static_cast<bool>((electronsIntegralValues_[4] + iOffset) % 2));
    }
    iEvent.put(electronsPutToken_, std::move(run3ScoutingElectrons));
  }

  void TestWriteRun3Scouting::produceMuons(edm::Event& iEvent) const {
    auto run3ScoutingMuons = std::make_unique<std::vector<Run3ScoutingMuon>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingMuons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      std::vector<int> vtxIndx;
      std::vector<uint16_t> hitPattern;
      vtxIndx.reserve(vectorSize);
      hitPattern.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        vtxIndx.push_back(static_cast<int>(muonsIntegralValues_[17] + iOffset + j * 10));
        hitPattern.push_back(static_cast<uint16_t>(muonsIntegralValues_[25] + iOffset + j * 10));
      }

      Run3ScoutingHitPatternPOD run3ScoutingHitPatternPOD;
      run3ScoutingHitPatternPOD.hitCount = static_cast<uint8_t>(muonsIntegralValues_[18] + iOffset);
      run3ScoutingHitPatternPOD.beginTrackHits = static_cast<uint8_t>(muonsIntegralValues_[19] + iOffset);
      run3ScoutingHitPatternPOD.endTrackHits = static_cast<uint8_t>(muonsIntegralValues_[20] + iOffset);
      run3ScoutingHitPatternPOD.beginInner = static_cast<uint8_t>(muonsIntegralValues_[21] + iOffset);
      run3ScoutingHitPatternPOD.endInner = static_cast<uint8_t>(muonsIntegralValues_[22] + iOffset);
      run3ScoutingHitPatternPOD.beginOuter = static_cast<uint8_t>(muonsIntegralValues_[23] + iOffset);
      run3ScoutingHitPatternPOD.endOuter = static_cast<uint8_t>(muonsIntegralValues_[24] + iOffset);
      run3ScoutingHitPatternPOD.hitPattern = std::move(hitPattern);

      run3ScoutingMuons->emplace_back(static_cast<float>(muonsFloatingPointValues_[0] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[1] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[2] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[3] + offset),
                                      static_cast<unsigned int>(muonsIntegralValues_[0] + iOffset),
                                      muonsIntegralValues_[1] + iOffset,
                                      static_cast<float>(muonsFloatingPointValues_[4] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[5] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[6] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[7] + offset),
                                      muonsIntegralValues_[2] + iOffset,
                                      muonsIntegralValues_[3] + iOffset,
                                      muonsIntegralValues_[4] + iOffset,
                                      muonsIntegralValues_[5] + iOffset,
                                      muonsIntegralValues_[6] + iOffset,
                                      muonsIntegralValues_[7] + iOffset,
                                      muonsIntegralValues_[8] + iOffset,
                                      static_cast<unsigned int>(muonsIntegralValues_[9] + iOffset),
                                      static_cast<unsigned int>(muonsIntegralValues_[10] + iOffset),
                                      muonsIntegralValues_[11] + iOffset,
                                      static_cast<unsigned int>(muonsIntegralValues_[12] + iOffset),
                                      muonsIntegralValues_[13] + iOffset,
                                      muonsIntegralValues_[14] + iOffset,
                                      muonsIntegralValues_[15] + iOffset,
                                      muonsIntegralValues_[16] + iOffset,
                                      static_cast<float>(muonsFloatingPointValues_[8] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[9] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[10] + offset),
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
                                      static_cast<float>(muonsFloatingPointValues_[23] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[24] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[25] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[26] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[27] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[28] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[29] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[30] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[31] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[32] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[33] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[34] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[35] + offset),
                                      static_cast<float>(muonsFloatingPointValues_[36] + offset),
                                      std::move(vtxIndx),
                                      std::move(run3ScoutingHitPatternPOD));
    }
    iEvent.put(muonsPutToken_, std::move(run3ScoutingMuons));
  }

  void TestWriteRun3Scouting::produceParticles(edm::Event& iEvent) const {
    auto run3ScoutingParticles = std::make_unique<std::vector<Run3ScoutingParticle>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingParticles->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);
      run3ScoutingParticles->emplace_back(static_cast<float>(particlesFloatingPointValues_[0] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[1] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[2] + offset),
                                          particlesIntegralValues_[0] + iOffset,
                                          particlesIntegralValues_[1] + iOffset,
                                          static_cast<float>(particlesFloatingPointValues_[3] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[4] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[5] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[6] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[7] + offset),
                                          static_cast<uint8_t>(particlesIntegralValues_[2] + iOffset),
                                          static_cast<uint8_t>(particlesIntegralValues_[3] + iOffset),
                                          static_cast<float>(particlesFloatingPointValues_[8] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[9] + offset),
                                          static_cast<float>(particlesFloatingPointValues_[10] + offset),
                                          static_cast<bool>((particlesIntegralValues_[4] + iOffset) % 2));
    }
    iEvent.put(particlesPutToken_, std::move(run3ScoutingParticles));
  }

  void TestWriteRun3Scouting::producePFJets(edm::Event& iEvent) const {
    auto run3ScoutingPFJets = std::make_unique<std::vector<Run3ScoutingPFJet>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingPFJets->reserve(vectorSize);

    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      std::vector<int> constituents;
      constituents.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        constituents.push_back(static_cast<int>(pfJetsIntegralValues_[7] + iOffset + j * 10));
      }

      run3ScoutingPFJets->emplace_back(static_cast<float>(pfJetsFloatingPointValues_[0] + offset),
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
    iEvent.put(pfJetsPutToken_, std::move(run3ScoutingPFJets));
  }

  void TestWriteRun3Scouting::producePhotons(edm::Event& iEvent) const {
    auto run3ScoutingPhotons = std::make_unique<std::vector<Run3ScoutingPhoton>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingPhotons->reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      std::vector<float> energyMatrix;
      std::vector<uint32_t> detIds;
      std::vector<float> timingMatrix;
      energyMatrix.reserve(vectorSize);
      detIds.reserve(vectorSize);
      timingMatrix.reserve(vectorSize);
      for (unsigned int j = 0; j < vectorSize; ++j) {
        energyMatrix.push_back(static_cast<float>(photonsFloatingPointValues_[12] + offset + j * 10));
        detIds.push_back(static_cast<uint32_t>(photonsIntegralValues_[1] + iOffset + j * 10));
        timingMatrix.push_back(static_cast<float>(photonsFloatingPointValues_[13] + offset + j * 10));
      }
      run3ScoutingPhotons->emplace_back(static_cast<float>(photonsFloatingPointValues_[0] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[1] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[2] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[3] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[4] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[5] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[6] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[7] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[8] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[9] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[10] + offset),
                                        static_cast<float>(photonsFloatingPointValues_[11] + offset),
                                        static_cast<uint32_t>(photonsIntegralValues_[0] + iOffset),
                                        std::move(energyMatrix),
                                        std::move(detIds),
                                        std::move(timingMatrix),
                                        static_cast<bool>((photonsIntegralValues_[2] + iOffset) % 2));
    }
    iEvent.put(photonsPutToken_, std::move(run3ScoutingPhotons));
  }

  void TestWriteRun3Scouting::produceTracks(edm::Event& iEvent) const {
    auto run3ScoutingTracks = std::make_unique<std::vector<Run3ScoutingTrack>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingTracks->reserve(vectorSize);

    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      run3ScoutingTracks->emplace_back(static_cast<float>(tracksFloatingPointValues_[0] + offset),
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
                                       static_cast<float>(tracksFloatingPointValues_[15] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[16] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[17] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[18] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[19] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[20] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[21] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[22] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[23] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[24] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[25] + offset),
                                       tracksIntegralValues_[4] + iOffset,
                                       static_cast<float>(tracksFloatingPointValues_[26] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[27] + offset),
                                       static_cast<float>(tracksFloatingPointValues_[28] + offset));
    }
    iEvent.put(tracksPutToken_, std::move(run3ScoutingTracks));
  }

  void TestWriteRun3Scouting::produceVertexes(edm::Event& iEvent) const {
    auto run3ScoutingVertexes = std::make_unique<std::vector<Run3ScoutingVertex>>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    run3ScoutingVertexes->reserve(vectorSize);

    for (unsigned int i = 0; i < vectorSize; ++i) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      run3ScoutingVertexes->emplace_back(static_cast<float>(vertexesFloatingPointValues_[0] + offset),
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
    iEvent.put(vertexesPutToken_, std::move(run3ScoutingVertexes));
  }

  void TestWriteRun3Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestWriteRun3Scouting constructor, test configuration error, " << msg;
  }

}  // namespace edmtest

using edmtest::TestWriteRun3Scouting;
DEFINE_FWK_MODULE(TestWriteRun3Scouting);
