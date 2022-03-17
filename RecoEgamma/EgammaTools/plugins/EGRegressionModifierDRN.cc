#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "RecoEgamma/EgammaTools/interface/EgammaRegressionContainer.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

/*
 * EGRegressionModifierDRN
 *
 * Object modifier to apply DRN regression.
 * Designed to be a drop-in replacement for EGRegressionModifierVX
 * 
 * Requires the appropriate DRNCorrectionProducerX(s) to also be in the path
 * You can specify which of reco::GsfElectron, reco::Photon, pat::Electron, pat::Photon 
 *    to apply corrections to in the config
 *
 */

class EGRegressionModifierDRN : public ModifyObjectValueBase {
public:
  EGRegressionModifierDRN(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);

  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;

  void modifyObject(reco::GsfElectron&) const final;
  void modifyObject(reco::Photon&) const final;

  void modifyObject(pat::Electron&) const final;
  void modifyObject(pat::Photon&) const final;

private:
  template <typename T>
  struct partVars {
    edm::InputTag source;
    edm::EDGetTokenT<edm::View<T>> token;
    const edm::View<T>* particles;

    edm::InputTag correctionsSource;
    edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> correctionsToken;
    const edm::ValueMap<std::pair<float, float>>* corrections;

    bool userFloat;
    std::string energyFloat, resFloat;

    unsigned i;

    partVars(const edm::ParameterSet& config, edm::ConsumesCollector& cc) {
      source = config.getParameter<edm::InputTag>("source");
      token = cc.consumes(source);

      correctionsSource = config.getParameter<edm::InputTag>("correctionsSource");
      correctionsToken = cc.consumes(correctionsSource);

      if (config.exists("energyFloat")) {
        userFloat = true;
        energyFloat = config.getParameter<std::string>("energyFloat");
        resFloat = config.getParameter<std::string>("resFloat");
      } else {
        userFloat = false;
      }

      i = 0;
    }

    const std::pair<float, float> getCorrection(T& part);

    const void doUserFloat(T& part, const std::pair<float, float>& correction) const {
      part.addUserFloat(energyFloat, correction.first);
      part.addUserFloat(resFloat, correction.second);
    }
  };

  std::unique_ptr<partVars<pat::Photon>> patPhotons_;
  std::unique_ptr<partVars<pat::Electron>> patElectrons_;
  std::unique_ptr<partVars<reco::Photon>> gedPhotons_;
  std::unique_ptr<partVars<reco::GsfElectron>> gsfElectrons_;
};

EGRegressionModifierDRN::EGRegressionModifierDRN(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf) {
  if (conf.exists("patPhotons")) {
    patPhotons_ = std::make_unique<partVars<pat::Photon>>(conf.getParameterSet("patPhotons"), cc);
  }

  if (conf.exists("gedPhotons")) {
    gedPhotons_ = std::make_unique<partVars<reco::Photon>>(conf.getParameterSet("gedPhotons"), cc);
  }

  if (conf.exists("patElectrons")) {
    patElectrons_ = std::make_unique<partVars<pat::Electron>>(conf.getParameterSet("patElectrons"), cc);
  }

  if (conf.exists("gsfElectrons")) {
    gsfElectrons_ = std::make_unique<partVars<reco::GsfElectron>>(conf.getParameterSet("gsfElectrons"), cc);
  }
}

void EGRegressionModifierDRN::setEvent(const edm::Event& evt) {
  if (patElectrons_) {
    patElectrons_->particles = &evt.get(patElectrons_->token);
    patElectrons_->corrections = &evt.get(patElectrons_->correctionsToken);
    patElectrons_->i = 0;
  }

  if (patPhotons_) {
    patPhotons_->particles = &evt.get(patPhotons_->token);
    patPhotons_->corrections = &evt.get(patPhotons_->correctionsToken);
    patPhotons_->i = 0;
  }

  if (gsfElectrons_) {
    gsfElectrons_->particles = &evt.get(gsfElectrons_->token);
    gsfElectrons_->corrections = &evt.get(gsfElectrons_->correctionsToken);
    gsfElectrons_->i = 0;
  }

  if (gedPhotons_) {
    gedPhotons_->particles = &evt.get(gedPhotons_->token);
    gedPhotons_->corrections = &evt.get(gedPhotons_->correctionsToken);
    gedPhotons_->i = 0;
  }
}

void EGRegressionModifierDRN::setEventContent(const edm::EventSetup& iSetup) {}

void EGRegressionModifierDRN::modifyObject(reco::GsfElectron& ele) const {
  if (!gsfElectrons_)
    return;

  const std::pair<float, float>& correction = gsfElectrons_->getCorrection(ele);

  if (correction.first > 0 && correction.second > 0) {
    ele.setCorrectedEcalEnergy(correction.first, true);
    ele.setCorrectedEcalEnergyError(correction.second);
  }

  throw cms::Exception("EGRegressionModifierDRN")
      << "Electron energy corrections not fully implemented yet:" << std::endl
      << "Still need E/p combination" << std::endl
      << "Do not enable DRN for electrons" << std::endl;
}

void EGRegressionModifierDRN::modifyObject(pat::Electron& ele) const {
  if (!patElectrons_)
    return;

  const std::pair<float, float>& correction = patElectrons_->getCorrection(ele);

  if (patElectrons_->userFloat) {
    patElectrons_->doUserFloat(ele, correction);
  } else if (correction.first > 0 && correction.second > 0) {
    ele.setCorrectedEcalEnergy(correction.first, true);
    ele.setCorrectedEcalEnergyError(correction.second);
  }

  throw cms::Exception("EGRegressionModifierDRN")
      << "Electron energy corrections not fully implemented yet:" << std::endl
      << "Still need E/p combination" << std::endl
      << "Do not enable DRN for electrons" << std::endl;
}

void EGRegressionModifierDRN::modifyObject(pat::Photon& pho) const {
  if (!patPhotons_)
    return;

  const std::pair<float, float>& correction = patPhotons_->getCorrection(pho);

  if (patPhotons_->userFloat) {
    patPhotons_->doUserFloat(pho, correction);
  } else if (correction.first > 0 && correction.second > 0) {
    pho.setCorrectedEnergy(pat::Photon::P4type::regression2, correction.first, correction.second, true);
  }
}

void EGRegressionModifierDRN::modifyObject(reco::Photon& pho) const {
  if (!gedPhotons_)
    return;

  const std::pair<float, float>& correction = gedPhotons_->getCorrection(pho);

  if (correction.first > 0 && correction.second > 0) {
    pho.setCorrectedEnergy(reco::Photon::P4type::regression2, correction.first, correction.second, true);
  }
};

template <typename T>
const std::pair<float, float> EGRegressionModifierDRN::partVars<T>::getCorrection(T& part) {
  edm::Ptr<T> ptr = particles->ptrAt(i++);

  std::pair<float, float> correction = (*corrections)[ptr];

  return correction;
}

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGRegressionModifierDRN, "EGRegressionModifierDRN");
