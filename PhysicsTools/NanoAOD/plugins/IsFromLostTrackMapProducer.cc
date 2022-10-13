// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      IsFromLostTrackMapProducer
//
/**\class IsFromLostTrackMapProducer IsFromLostTrackMapProducer.cc PhysicsTools/NanoAOD/plugins/IsFromLostTrackMapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Maria Giulia Ratti (ETHZ) [mratti]
//         Created:  Thu, 22 Nov 2018 12:34:48 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//
// class declaration
//

class IsFromLostTrackMapProducer : public edm::global::EDProducer<> {
public:
  explicit IsFromLostTrackMapProducer(const edm::ParameterSet& iConfig)
      : srcIsoTracks_(consumes<edm::View<pat::IsolatedTrack>>(
            iConfig.getParameter<edm::InputTag>("srcIsoTracks"))),  // final isolated tracks
        lt_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks")))  // lost tracks
  {
    produces<edm::ValueMap<bool>>("isFromLostTrack");  // name of the value map that I want to actually produce
  }
  ~IsFromLostTrackMapProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<pat::IsolatedTrack>> srcIsoTracks_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> lt_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

// ------------ method called to produce the data  ------------
void IsFromLostTrackMapProducer::produce(edm::StreamID streamID,
                                         edm::Event& iEvent,
                                         const edm::EventSetup& iSetup) const {
  // isolated tracks
  auto srcIsoTracks = iEvent.getHandle(srcIsoTracks_);

  // lostTracks collection
  auto ltHandle = iEvent.getHandle(lt_);

  // the map cannot be filled straight away, so create an intermediate vector
  unsigned int Nit = srcIsoTracks->size();
  std::vector<bool> v_isFromLostTrack(Nit, false);

  for (unsigned int iit = 0; iit < Nit; iit++) {
    auto isotrack = srcIsoTracks->ptrAt(iit);
    pat::PackedCandidateRef pcref =
        isotrack->packedCandRef();  // this is either the reference to the pf candidate or to the lost track
    bool isFromLostTrack = (pcref.isNonnull() && pcref.id() == ltHandle.id());
    v_isFromLostTrack[iit] = isFromLostTrack;
  }

  auto vm_isFromLostTrack = std::make_unique<edm::ValueMap<bool>>();
  edm::ValueMap<bool>::Filler filler(*vm_isFromLostTrack);
  filler.insert(srcIsoTracks, v_isFromLostTrack.begin(), v_isFromLostTrack.end());
  filler.fill();
  iEvent.put(std::move(vm_isFromLostTrack), "isFromLostTrack");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsFromLostTrackMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcIsoTracks")->setComment("isolated track input collection");
  desc.add<edm::InputTag>("lostTracks")->setComment("lost tracks collection");

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsFromLostTrackMapProducer);
