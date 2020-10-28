// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      ProtonProducer
//
/**\class ProtonProducer ProtonProducer.cc PhysicsTools/NanoAOD/plugins/ProtonProducer.cc
 Description: Realavent proton variables for analysis usage
 Implementation:
*/
//
// Original Author:  Justin Williams
//         Created: 04 Jul 2019 15:27:53 GMT
//
//

// system include files
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CommonTools/Egamma/interface/EffectiveAreas.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

class ProtonProducer : public edm::global::EDProducer<> {
public:
  ProtonProducer(edm::ParameterSet const &ps)
      : tokenRecoProtonsSingleRP_(
            mayConsume<reco::ForwardProtonCollection>(ps.getParameter<edm::InputTag>("tagRecoProtonsSingle"))),
        tokenRecoProtonsMultiRP_(
            mayConsume<reco::ForwardProtonCollection>(ps.getParameter<edm::InputTag>("tagRecoProtonsMulti"))),
        tokenTracksLite_(mayConsume<std::vector<CTPPSLocalTrackLite>>(ps.getParameter<edm::InputTag>("tagTrackLite"))) {
    produces<edm::ValueMap<int>>("protonRPId");
    produces<edm::ValueMap<int>>("arm");
    produces<nanoaod::FlatTable>("ppsTrackTable");
  }
  ~ProtonProducer() override {}

  // ------------ method called to produce the data  ------------
  void produce(edm::StreamID id, edm::Event &iEvent, const edm::EventSetup &iSetup) const override {
    // Get Forward Proton handles
    edm::Handle<reco::ForwardProtonCollection> hRecoProtonsSingleRP;
    iEvent.getByToken(tokenRecoProtonsSingleRP_, hRecoProtonsSingleRP);

    edm::Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
    iEvent.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

    // Get PPS Local Track handle
    edm::Handle<std::vector<CTPPSLocalTrackLite>> ppsTracksLite;
    iEvent.getByToken(tokenTracksLite_, ppsTracksLite);

    // book output variables for protons
    std::vector<int> singleRP_RPId;
    std::vector<int> multiRP_arm;

    // book output variables for tracks
    std::vector<float> trackX, trackY, trackTime, trackTimeUnc, localSlopeX, localSlopeY, normalizedChi2;
    std::vector<int> singleRPProtonIdx, multiRPProtonIdx, decRPId, numFitPoints, pixelRecoInfo, rpType;

    // process single-RP protons
    {
      const auto &num_proton = hRecoProtonsSingleRP->size();
      singleRP_RPId.reserve(num_proton);

      for (const auto &proton : *hRecoProtonsSingleRP) {
        CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
        unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
        singleRP_RPId.push_back(rpDecId);
      }
    }

    // process multi-RP protons
    {
      const auto &num_proton = hRecoProtonsMultiRP->size();
      multiRP_arm.reserve(num_proton);

      for (const auto &proton : *hRecoProtonsMultiRP) {
        multiRP_arm.push_back((proton.pz() < 0.) ? 1 : 0);
      }
    }

    // process local tracks
    for (unsigned int tr_idx = 0; tr_idx < ppsTracksLite->size(); ++tr_idx) {
      const auto &tr = ppsTracksLite->at(tr_idx);

      CTPPSDetId rpId(tr.rpId());
      unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      decRPId.push_back(rpDecId);
      rpType.push_back(rpId.subdetId());

      trackX.push_back(tr.x());
      trackY.push_back(tr.y());
      trackTime.push_back(tr.time());
      trackTimeUnc.push_back(tr.timeUnc());
      numFitPoints.push_back(tr.numberOfPointsUsedForFit());
      pixelRecoInfo.push_back(static_cast<int>(tr.pixelTrackRecoInfo()));
      normalizedChi2.push_back(tr.chiSquaredOverNDF());
      localSlopeX.push_back(tr.tx());
      localSlopeY.push_back(tr.ty());

      signed int singleRP_idx = -1;
      for (unsigned int p_idx = 0; p_idx < hRecoProtonsSingleRP->size(); ++p_idx) {
        const auto &proton = hRecoProtonsSingleRP->at(p_idx);

        for (const auto &ref : proton.contributingLocalTracks()) {
          if (ref.key() == tr_idx)
            singleRP_idx = p_idx;
        }
      }
      singleRPProtonIdx.push_back(singleRP_idx);

      signed int multiRP_idx = -1;
      for (unsigned int p_idx = 0; p_idx < hRecoProtonsMultiRP->size(); ++p_idx) {
        const auto &proton = hRecoProtonsMultiRP->at(p_idx);

        for (const auto &ref : proton.contributingLocalTracks()) {
          if (ref.key() == tr_idx)
            multiRP_idx = p_idx;
        }
      }
      multiRPProtonIdx.push_back(multiRP_idx);
    }

    // update proton tables
    std::unique_ptr<edm::ValueMap<int>> protonRPIdV(new edm::ValueMap<int>());
    edm::ValueMap<int>::Filler fillerID(*protonRPIdV);
    fillerID.insert(hRecoProtonsSingleRP, singleRP_RPId.begin(), singleRP_RPId.end());
    fillerID.fill();

    // Testing
    std::unique_ptr<edm::ValueMap<int>> multiRP_armV(new edm::ValueMap<int>());
    edm::ValueMap<int>::Filler fillermultiArm(*multiRP_armV);
    fillermultiArm.insert(hRecoProtonsMultiRP, multiRP_arm.begin(), multiRP_arm.end());
    fillermultiArm.fill();

    // build track table
    auto ppsTab = std::make_unique<nanoaod::FlatTable>(trackX.size(), "PPSLocalTrack", false);
    ppsTab->addColumn<int>("singleRPProtonIdx", singleRPProtonIdx, "local track - proton correspondence");
    ppsTab->addColumn<int>("multiRPProtonIdx", multiRPProtonIdx, "local track - proton correspondence");
    ppsTab->addColumn<float>("x", trackX, "local track x", 16);
    ppsTab->addColumn<float>("y", trackY, "local track y", 13);
    ppsTab->addColumn<float>("time", trackTime, "local track time", 16);
    ppsTab->addColumn<float>("timeUnc", trackTimeUnc, "local track time uncertainty", 13);
    ppsTab->addColumn<int>("decRPId", decRPId, "local track detector dec id");
    ppsTab->addColumn<int>("rpType", rpType, "strip=3, pixel=4, diamond=5, timing=6");
    ppsTab->setDoc("ppsLocalTrack variables");

    // save output
    iEvent.put(std::move(protonRPIdV), "protonRPId");
    iEvent.put(std::move(multiRP_armV), "arm");
    iEvent.put(std::move(ppsTab), "ppsTrackTable");
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tagRecoProtonsSingle")->setComment("singleRP proton collection");
    desc.add<edm::InputTag>("tagRecoProtonsMulti")->setComment("multiRP proton collection");
    desc.add<edm::InputTag>("tagTrackLite")->setComment("pps local tracks lite collection");
    descriptions.add("ProtonProducer", desc);
  }

protected:
  const edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsSingleRP_;
  const edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;
  const edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite>> tokenTracksLite_;
};

DEFINE_FWK_MODULE(ProtonProducer);
