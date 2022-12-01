// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      IsoValueMapProducer
//
/**\class IsoValueMapProducer IsoValueMapProducer.cc PhysicsTools/NanoAOD/plugins/IsoValueMapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Peruzzi
//         Created:  Mon, 04 Sep 2017 22:43:53 GMT
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
class IsoValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit IsoValueMapProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        relative_(iConfig.getParameter<bool>("relative")),
        doQuadratic_(iConfig.getParameter<bool>("doQuadratic")) {
    if ((typeid(T) == typeid(pat::Muon)) || (typeid(T) == typeid(pat::Electron)) ||
        typeid(T) == typeid(pat::IsolatedTrack)) {
      produces<edm::ValueMap<float>>("miniIsoChg");
      produces<edm::ValueMap<float>>("miniIsoAll");
      ea_miniiso_ =
          std::make_unique<EffectiveAreas>((iConfig.getParameter<edm::FileInPath>("EAFile_MiniIso")).fullPath());
      rho_miniiso_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho_MiniIso"));
    }
    if ((typeid(T) == typeid(pat::Electron))) {
      produces<edm::ValueMap<float>>("PFIsoChg");
      produces<edm::ValueMap<float>>("PFIsoAll");
      produces<edm::ValueMap<float>>("PFIsoAll04");
      ea_pfiso_ = std::make_unique<EffectiveAreas>((iConfig.getParameter<edm::FileInPath>("EAFile_PFIso")).fullPath());
      rho_pfiso_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho_PFIso"));
    } else if ((typeid(T) == typeid(pat::Photon))) {
      rho_pfiso_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho_PFIso"));

      if (!doQuadratic_) {
        produces<edm::ValueMap<float>>("PFIsoChg");
        produces<edm::ValueMap<float>>("PFIsoAll");

        ea_pfiso_chg_ =
            std::make_unique<EffectiveAreas>((iConfig.getParameter<edm::FileInPath>("EAFile_PFIso_Chg")).fullPath());
        ea_pfiso_neu_ =
            std::make_unique<EffectiveAreas>((iConfig.getParameter<edm::FileInPath>("EAFile_PFIso_Neu")).fullPath());
        ea_pfiso_pho_ =
            std::make_unique<EffectiveAreas>((iConfig.getParameter<edm::FileInPath>("EAFile_PFIso_Pho")).fullPath());

      }

      else {
        produces<edm::ValueMap<float>>("PFIsoChgQuadratic");
        produces<edm::ValueMap<float>>("PFIsoAllQuadratic");

        quadratic_ea_pfiso_chg_ = std::make_unique<EffectiveAreas>(
            (iConfig.getParameter<edm::FileInPath>("QuadraticEAFile_PFIso_Chg")).fullPath(), true);
        quadratic_ea_pfiso_ecal_ = std::make_unique<EffectiveAreas>(
            (iConfig.getParameter<edm::FileInPath>("QuadraticEAFile_PFIso_ECal")).fullPath(), true);
        quadratic_ea_pfiso_hcal_ = std::make_unique<EffectiveAreas>(
            (iConfig.getParameter<edm::FileInPath>("QuadraticEAFile_PFIso_HCal")).fullPath(), true);
      }
    }
  }

  ~IsoValueMapProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  bool relative_;
  bool doQuadratic_;
  edm::EDGetTokenT<double> rho_miniiso_;
  edm::EDGetTokenT<double> rho_pfiso_;
  std::unique_ptr<EffectiveAreas> ea_miniiso_;
  std::unique_ptr<EffectiveAreas> ea_pfiso_;
  std::unique_ptr<EffectiveAreas> ea_pfiso_chg_;
  std::unique_ptr<EffectiveAreas> ea_pfiso_neu_;
  std::unique_ptr<EffectiveAreas> ea_pfiso_pho_;
  std::unique_ptr<EffectiveAreas> quadratic_ea_pfiso_chg_;
  std::unique_ptr<EffectiveAreas> quadratic_ea_pfiso_ecal_;
  std::unique_ptr<EffectiveAreas> quadratic_ea_pfiso_hcal_;

  float getEtaForEA(const T*) const;
  void doMiniIso(edm::Event&) const;
  void doPFIsoEle(edm::Event&) const;
  void doPFIsoPho(edm::Event&) const;
  void doPFIsoPhoQuadratic(edm::Event&) const;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

template <typename T>
float IsoValueMapProducer<T>::getEtaForEA(const T* obj) const {
  return obj->eta();
}
template <>
float IsoValueMapProducer<pat::Electron>::getEtaForEA(const pat::Electron* el) const {
  return el->superCluster()->eta();
}
template <>
float IsoValueMapProducer<pat::Photon>::getEtaForEA(const pat::Photon* ph) const {
  return ph->superCluster()->eta();
}

template <typename T>
void IsoValueMapProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if ((typeid(T) == typeid(pat::Muon)) || (typeid(T) == typeid(pat::Electron)) ||
      typeid(T) == typeid(pat::IsolatedTrack)) {
    doMiniIso(iEvent);
  };
  if ((typeid(T) == typeid(pat::Electron))) {
    doPFIsoEle(iEvent);
  }
  if ((typeid(T) == typeid(pat::Photon))) {
    if (!doQuadratic_)
      doPFIsoPho(iEvent);
    else
      doPFIsoPhoQuadratic(iEvent);
  }
}

template <typename T>
void IsoValueMapProducer<T>::doMiniIso(edm::Event& iEvent) const {
  auto src = iEvent.getHandle(src_);
  const auto& rho = iEvent.get(rho_miniiso_);

  unsigned int nInput = src->size();

  std::vector<float> miniIsoChg, miniIsoAll;
  miniIsoChg.reserve(nInput);
  miniIsoAll.reserve(nInput);

  for (const auto& obj : *src) {
    auto iso = obj.miniPFIsolation();
    auto chg = iso.chargedHadronIso();
    auto neu = iso.neutralHadronIso();
    auto pho = iso.photonIso();
    auto ea = ea_miniiso_->getEffectiveArea(fabs(getEtaForEA(&obj)));
    float R = 10.0 / std::min(std::max(obj.pt(), 50.0), 200.0);
    ea *= std::pow(R / 0.3, 2);
    float scale = relative_ ? 1.0 / obj.pt() : 1;
    miniIsoChg.push_back(scale * chg);
    miniIsoAll.push_back(scale * (chg + std::max(0.0, neu + pho - rho * ea)));
  }

  auto miniIsoChgV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerChg(*miniIsoChgV);
  fillerChg.insert(src, miniIsoChg.begin(), miniIsoChg.end());
  fillerChg.fill();
  auto miniIsoAllV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerAll(*miniIsoAllV);
  fillerAll.insert(src, miniIsoAll.begin(), miniIsoAll.end());
  fillerAll.fill();

  iEvent.put(std::move(miniIsoChgV), "miniIsoChg");
  iEvent.put(std::move(miniIsoAllV), "miniIsoAll");
}

template <>
void IsoValueMapProducer<pat::Photon>::doMiniIso(edm::Event& iEvent) const {}

template <typename T>
void IsoValueMapProducer<T>::doPFIsoEle(edm::Event& iEvent) const {}

template <>
void IsoValueMapProducer<pat::Electron>::doPFIsoEle(edm::Event& iEvent) const {
  edm::Handle<edm::View<pat::Electron>> src;
  iEvent.getByToken(src_, src);
  const auto& rho = iEvent.get(rho_pfiso_);

  unsigned int nInput = src->size();

  std::vector<float> PFIsoChg, PFIsoAll, PFIsoAll04;
  PFIsoChg.reserve(nInput);
  PFIsoAll.reserve(nInput);
  PFIsoAll04.reserve(nInput);

  for (const auto& obj : *src) {
    auto iso = obj.pfIsolationVariables();
    auto chg = iso.sumChargedHadronPt;
    auto neu = iso.sumNeutralHadronEt;
    auto pho = iso.sumPhotonEt;
    auto ea = ea_pfiso_->getEffectiveArea(fabs(getEtaForEA(&obj)));
    float scale = relative_ ? 1.0 / obj.pt() : 1;
    PFIsoChg.push_back(scale * chg);
    PFIsoAll.push_back(scale * (chg + std::max(0.0, neu + pho - rho * ea)));
    PFIsoAll04.push_back(scale * (obj.chargedHadronIso() +
                                  std::max(0.0, obj.neutralHadronIso() + obj.photonIso() - rho * ea * 16. / 9.)));
  }

  auto PFIsoChgV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerChg(*PFIsoChgV);
  fillerChg.insert(src, PFIsoChg.begin(), PFIsoChg.end());
  fillerChg.fill();
  auto PFIsoAllV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerAll(*PFIsoAllV);
  fillerAll.insert(src, PFIsoAll.begin(), PFIsoAll.end());
  fillerAll.fill();
  auto PFIsoAll04V = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerAll04(*PFIsoAll04V);
  fillerAll04.insert(src, PFIsoAll04.begin(), PFIsoAll04.end());
  fillerAll04.fill();

  iEvent.put(std::move(PFIsoChgV), "PFIsoChg");
  iEvent.put(std::move(PFIsoAllV), "PFIsoAll");
  iEvent.put(std::move(PFIsoAll04V), "PFIsoAll04");
}

template <typename T>
void IsoValueMapProducer<T>::doPFIsoPho(edm::Event& iEvent) const {}

template <>
void IsoValueMapProducer<pat::Photon>::doPFIsoPho(edm::Event& iEvent) const {
  edm::Handle<edm::View<pat::Photon>> src;
  iEvent.getByToken(src_, src);
  const auto& rho = iEvent.get(rho_pfiso_);

  unsigned int nInput = src->size();

  std::vector<float> PFIsoChg, PFIsoAll;

  PFIsoChg.reserve(nInput);
  PFIsoAll.reserve(nInput);

  for (const auto& obj : *src) {
    auto chg = obj.chargedHadronIso();
    auto neu = obj.neutralHadronIso();
    auto pho = obj.photonIso();

    auto ea_chg = ea_pfiso_chg_->getEffectiveArea(fabs(getEtaForEA(&obj)));
    auto ea_neu = ea_pfiso_neu_->getEffectiveArea(fabs(getEtaForEA(&obj)));
    auto ea_pho = ea_pfiso_pho_->getEffectiveArea(fabs(getEtaForEA(&obj)));

    float scale = relative_ ? 1.0 / obj.pt() : 1;
    PFIsoChg.push_back(scale * std::max(0.0, chg - rho * ea_chg));
    PFIsoAll.push_back(PFIsoChg.back() +
                       scale * (std::max(0.0, neu - rho * ea_neu) + std::max(0.0, pho - rho * ea_pho)));
  }

  auto PFIsoChgV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerChg(*PFIsoChgV);
  fillerChg.insert(src, PFIsoChg.begin(), PFIsoChg.end());
  fillerChg.fill();
  auto PFIsoAllV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerAll(*PFIsoAllV);
  fillerAll.insert(src, PFIsoAll.begin(), PFIsoAll.end());
  fillerAll.fill();

  iEvent.put(std::move(PFIsoChgV), "PFIsoChg");
  iEvent.put(std::move(PFIsoAllV), "PFIsoAll");
}

template <typename T>
void IsoValueMapProducer<T>::doPFIsoPhoQuadratic(edm::Event& iEvent) const {}

template <>
void IsoValueMapProducer<pat::Photon>::doPFIsoPhoQuadratic(edm::Event& iEvent) const {
  edm::Handle<edm::View<pat::Photon>> src;
  iEvent.getByToken(src_, src);
  const auto& rho = iEvent.get(rho_pfiso_);

  unsigned int nInput = src->size();

  std::vector<float> PFIsoChgQuadratic, PFIsoAllQuadratic;

  PFIsoChgQuadratic.reserve(nInput);
  PFIsoAllQuadratic.reserve(nInput);

  for (const auto& obj : *src) {
    auto chg = obj.chargedHadronIso();
    auto ecal = obj.ecalPFClusterIso();
    auto hcal = obj.hcalPFClusterIso();

    auto quadratic_ea_chg = quadratic_ea_pfiso_chg_->getQuadraticEA(fabs(getEtaForEA(&obj)));
    auto linear_ea_chg = quadratic_ea_pfiso_chg_->getLinearEA(fabs(getEtaForEA(&obj)));
    auto quadratic_ea_ecal = quadratic_ea_pfiso_ecal_->getQuadraticEA(fabs(getEtaForEA(&obj)));
    auto linear_ea_ecal = quadratic_ea_pfiso_ecal_->getLinearEA(fabs(getEtaForEA(&obj)));
    auto quadratic_ea_hcal = quadratic_ea_pfiso_hcal_->getQuadraticEA(fabs(getEtaForEA(&obj)));
    auto linear_ea_hcal = quadratic_ea_pfiso_hcal_->getLinearEA(fabs(getEtaForEA(&obj)));

    float scale = relative_ ? 1.0 / obj.pt() : 1;

    PFIsoChgQuadratic.push_back(scale * std::max(0.0, chg - (quadratic_ea_chg * rho * rho + linear_ea_chg * rho)));
    PFIsoAllQuadratic.push_back(PFIsoChgQuadratic.back() +
                                scale * (std::max(0.0, ecal - (quadratic_ea_ecal * rho * rho + linear_ea_ecal * rho)) +
                                         std::max(0.0, hcal - (quadratic_ea_hcal * rho * rho + linear_ea_hcal * rho))));
  }

  auto PFIsoChgQuadraticV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerChgQuadratic(*PFIsoChgQuadraticV);
  fillerChgQuadratic.insert(src, PFIsoChgQuadratic.begin(), PFIsoChgQuadratic.end());
  fillerChgQuadratic.fill();

  auto PFIsoAllQuadraticV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerAllQuadratic(*PFIsoAllQuadraticV);
  fillerAllQuadratic.insert(src, PFIsoAllQuadratic.begin(), PFIsoAllQuadratic.end());
  fillerAllQuadratic.fill();

  iEvent.put(std::move(PFIsoChgQuadraticV), "PFIsoChgQuadratic");
  iEvent.put(std::move(PFIsoAllQuadraticV), "PFIsoAllQuadratic");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void IsoValueMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<bool>("relative")->setComment("compute relative isolation instead of absolute one");
  desc.add<bool>("doQuadratic", false)->setComment("flag to do quadratic EA corections for photons");
  if ((typeid(T) == typeid(pat::Muon)) || (typeid(T) == typeid(pat::Electron)) ||
      typeid(T) == typeid(pat::IsolatedTrack)) {
    desc.add<edm::FileInPath>("EAFile_MiniIso")
        ->setComment("txt file containing effective areas to be used for mini-isolation pileup subtraction");
    desc.add<edm::InputTag>("rho_MiniIso")
        ->setComment("rho to be used for effective-area based mini-isolation pileup subtraction");
  }
  if ((typeid(T) == typeid(pat::Electron))) {
    desc.add<edm::FileInPath>("EAFile_PFIso")
        ->setComment(
            "txt file containing effective areas to be used for PF-isolation pileup subtraction for electrons");
    desc.add<edm::InputTag>("rho_PFIso")
        ->setComment("rho to be used for effective-area based PF-isolation pileup subtraction for electrons");
  }
  if ((typeid(T) == typeid(pat::Photon))) {
    desc.addOptional<edm::InputTag>("mapIsoChg")
        ->setComment("input charged PF isolation calculated in VID for photons");
    desc.addOptional<edm::InputTag>("mapIsoNeu")
        ->setComment("input neutral PF isolation calculated in VID for photons");
    desc.addOptional<edm::InputTag>("mapIsoPho")->setComment("input photon PF isolation calculated in VID for photons");

    desc.addOptional<edm::FileInPath>("EAFile_PFIso_Chg")
        ->setComment(
            "txt file containing effective areas to be used for charged PF-isolation pileup subtraction for photons");
    desc.addOptional<edm::FileInPath>("EAFile_PFIso_Neu")
        ->setComment(
            "txt file containing effective areas to be used for neutral PF-isolation pileup subtraction for photons");
    desc.addOptional<edm::FileInPath>("EAFile_PFIso_Pho")
        ->setComment(
            "txt file containing effective areas to be used for photon PF-isolation pileup subtraction for photons");

    desc.add<edm::InputTag>("rho_PFIso")
        ->setComment("rho to be used for effective-area based PF-isolation pileup subtraction for photons");

    desc.addOptional<edm::FileInPath>("QuadraticEAFile_PFIso_Chg")
        ->setComment(
            "txt file containing quadratic effective areas to be used for charged PF-isolation pileup subtraction for "
            "photons");
    desc.addOptional<edm::FileInPath>("QuadraticEAFile_PFIso_ECal")
        ->setComment(
            "txt file containing quadratic effective areas to be used for ecal PF-isolation pileup subtraction for "
            "photons");
    desc.addOptional<edm::FileInPath>("QuadraticEAFile_PFIso_HCal")
        ->setComment(
            "txt file containing quadratic effective areas to be used for hcal PF-isolation pileup subtraction for "
            "photons");
  }

  std::string modname;
  if (typeid(T) == typeid(pat::Muon))
    modname += "Muon";
  else if (typeid(T) == typeid(pat::Electron))
    modname += "Ele";
  else if (typeid(T) == typeid(pat::Photon))
    modname += "Pho";
  else if (typeid(T) == typeid(pat::IsolatedTrack))
    modname += "IsoTrack";
  modname += "IsoValueMapProducer";
  descriptions.add(modname, desc);
}

typedef IsoValueMapProducer<pat::Muon> MuonIsoValueMapProducer;
typedef IsoValueMapProducer<pat::Electron> EleIsoValueMapProducer;
typedef IsoValueMapProducer<pat::Photon> PhoIsoValueMapProducer;
typedef IsoValueMapProducer<pat::IsolatedTrack> IsoTrackIsoValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIsoValueMapProducer);
DEFINE_FWK_MODULE(EleIsoValueMapProducer);
DEFINE_FWK_MODULE(PhoIsoValueMapProducer);
DEFINE_FWK_MODULE(IsoTrackIsoValueMapProducer);
