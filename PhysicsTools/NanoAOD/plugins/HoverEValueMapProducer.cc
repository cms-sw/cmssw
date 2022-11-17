// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      HoverEValueMapProducer
//
/**\class HoverEValueMapProducer HoverEValueMapProducer.cc PhysicsTools/NanoAOD/plugins/HoverEValueMapProducer.cc

 Description: This class implements the quadratic EA correction for H/E variable in cutBasedPhotonID.
              This class is implemented following Marco Peruzzi's IsoValueMapProducer class

 Implementation:
     [Notes on implementation]
*/
//
//          Author:  Shubham Dutta
//         Created:  Tue, 01 Nov 2022 07:45 IST
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

#include "CommonTools/Egamma/interface/EffectiveAreas.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"

//
// class declaration
//

template <typename T>
class HoverEValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit HoverEValueMapProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        relative_(iConfig.getParameter<bool>("relative")) {
    if ((typeid(T) == typeid(pat::Photon))) {
      produces<edm::ValueMap<float>>("HoEForPhoEACorr");

      rho_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho"));

      quadratic_ea_hOverE_ = std::make_unique<EffectiveAreas>(
          (iConfig.getParameter<edm::FileInPath>("QuadraticEAFile_HoverE")).fullPath(), true);
    }
  }
  ~HoverEValueMapProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  bool relative_;
  edm::EDGetTokenT<double> rho_;
  std::unique_ptr<EffectiveAreas> quadratic_ea_hOverE_;
  float getEtaForEA(const T*) const;
  void doHoverEPho(edm::Event&) const;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

template <typename T>
float HoverEValueMapProducer<T>::getEtaForEA(const T* obj) const {
  return obj->eta();
}

template <>
float HoverEValueMapProducer<pat::Photon>::getEtaForEA(const pat::Photon* ph) const {
  return ph->superCluster()->eta();
}

template <typename T>
void HoverEValueMapProducer<T>::produce(edm::StreamID streamID,
                                        edm::Event& iEvent,
                                        const edm::EventSetup& iSetup) const {
  if ((typeid(T) == typeid(pat::Photon))) {
    doHoverEPho(iEvent);
  }
}

template <typename T>
void HoverEValueMapProducer<T>::doHoverEPho(edm::Event& iEvent) const {}

template <>
void HoverEValueMapProducer<pat::Photon>::doHoverEPho(edm::Event& iEvent) const {
  auto src = iEvent.getHandle(src_);
  const auto& rho = iEvent.get(rho_);

  unsigned int nInput = src->size();

  std::vector<float> HoverEQuadratic;
  HoverEQuadratic.reserve(nInput);

  for (const auto& obj : *src) {
    auto hOverE = obj.hcalOverEcal();

    auto quadratic_ea_hOverE = quadratic_ea_hOverE_->getQuadraticEA(fabs(getEtaForEA(&obj)));
    auto linear_ea_hOverE = quadratic_ea_hOverE_->getLinearEA(fabs(getEtaForEA(&obj)));

    float scale = relative_ ? 1.0 / obj.pt() : 1;

    HoverEQuadratic.push_back(scale *
                              (std::max(0.0, hOverE - (quadratic_ea_hOverE * rho * rho + linear_ea_hOverE * rho))));
  }

  std::unique_ptr<edm::ValueMap<float>> HoverEQuadraticV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerHoverEQuadratic(*HoverEQuadraticV);
  fillerHoverEQuadratic.insert(src, HoverEQuadratic.begin(), HoverEQuadratic.end());
  fillerHoverEQuadratic.fill();

  iEvent.put(std::move(HoverEQuadraticV), "HoEForPhoEACorr");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void HoverEValueMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<bool>("relative")->setComment("compute relative HoverE instead of absolute one");
  if ((typeid(T) == typeid(pat::Photon))) {
    desc.add<edm::FileInPath>("QuadraticEAFile_HoverE")
        ->setComment("txt file containing quadratic effective areas to be used for H/E pileup subtraction for photons");

    desc.add<edm::InputTag>("rho")->setComment(
        "rho to be used for effective-area based H/E pileup subtraction for photons");
  }

  //  std::string modname;
  //  if (typeid(T) == typeid(pat::Photon))
  //    modname += "Pho";
  //  modname += "HoverEValueMapProducer";
  descriptions.addWithDefaultLabel(desc);
}

typedef HoverEValueMapProducer<pat::Photon> PhoHoverEValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(PhoHoverEValueMapProducer);
