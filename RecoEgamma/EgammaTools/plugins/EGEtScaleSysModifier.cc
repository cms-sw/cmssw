#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"

#include <vdt/vdtMath.h>

//in the legacy re-reco we came across an Et scale issue, specifically an inflection at 45 GeV
//it is problematic to address in the normal scale and smearing code therefore
//this patch modifies the e/gamma object to "patch in" this additional systematic
//Questions:
//1 ) Why dont we add this to the overall scale systematic?
//Well we could but the whole point of this is to get a proper template which can be evolved
//correctly. The issue is not that the current systematic doesnt cover this systematic on a
//per bin basis, it does, the problem is the sign flips at 45 GeV so its fine if you use
//only eles/phos > 45 or <45 GeV but the template cant simultaneously cover the entire spectrum
//and you'll get a nasty constrained fit.
//And if you care about these sort of things (and you probably should), you shouldnt be using
//the overall systematic anyways but its seperate parts
//
//2 ) Why dont you do it inside the standard class
//We could but we're rather hoping to solve this issue more cleanly in the future
//but we would hold up the reminiAOD too long to do so so this is just to get something
//which should be fine for 90% of analyses in the miniAOD. So this is a temporary fix
//which we hope to rip out soon (if you're reading this in 2024 because we're still doing it
//this way, all I can say is sorry, we thought it would go away soon! )

class EGEtScaleSysModifier : public ModifyObjectValueBase {
public:
  EGEtScaleSysModifier(const edm::ParameterSet& conf, edm::ConsumesCollector&);
  ~EGEtScaleSysModifier() override {}

  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;

  void modifyObject(pat::Electron& ele) const final;
  void modifyObject(pat::Photon& pho) const final;

private:
  std::pair<float, float> calCombinedMom(reco::GsfElectron& ele, const float scale, const float smear) const;
  void setEcalEnergy(reco::GsfElectron& ele, const float scale, const float smear) const;

  class UncertFuncBase {
  public:
    UncertFuncBase() {}
    virtual ~UncertFuncBase() {}
    virtual float val(const float et) const = 0;
  };

  //defines two uncertaintes, one Et<X and one Et>Y
  //for X<Et<Y, it linearly extrapolates betwen the two values
  class UncertFuncV1 : public UncertFuncBase {
  public:
    UncertFuncV1(const edm::ParameterSet& conf)
        : lowEt_(conf.getParameter<double>("lowEt")),
          highEt_(conf.getParameter<double>("highEt")),
          lowEtUncert_(conf.getParameter<double>("lowEtUncert")),
          highEtUncert_(conf.getParameter<double>("highEtUncert")),
          dEt_(highEt_ - lowEt_),
          dUncert_(highEtUncert_ - lowEtUncert_) {
      if (highEt_ <= lowEt_)
        throw cms::Exception("ConfigError") << " highEt " << highEt_ << " is not higher than lowEt " << lowEt_;
    }
    ~UncertFuncV1() override {}

    float val(const float et) const override {
      if (et <= lowEt_)
        return lowEtUncert_;
      else if (et >= highEt_)
        return highEtUncert_;
      else {
        return (et - lowEt_) * dUncert_ / dEt_ + lowEtUncert_;
      }
    }

  private:
    float lowEt_;
    float highEt_;
    float lowEtUncert_;
    float highEtUncert_;
    float dEt_;
    float dUncert_;
  };

  EpCombinationTool epCombTool_;
  std::unique_ptr<UncertFuncBase> uncertFunc_;
};

EGEtScaleSysModifier::EGEtScaleSysModifier(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf), epCombTool_{conf.getParameterSet("epCombConfig"), std::move(cc)} {
  const edm::ParameterSet& funcPSet = conf.getParameterSet("uncertFunc");
  const std::string& funcName = funcPSet.getParameter<std::string>("name");
  if (funcName == "UncertFuncV1") {
    uncertFunc_ = std::make_unique<UncertFuncV1>(funcPSet);
  } else {
    throw cms::Exception("ConfigError") << "Error constructing EGEtScaleSysModifier, function name " << funcName
                                        << " not valid";
  }
}

void EGEtScaleSysModifier::setEvent(const edm::Event& iEvent) {}

void EGEtScaleSysModifier::setEventContent(const edm::EventSetup& iSetup) { epCombTool_.setEventContent(iSetup); }

void EGEtScaleSysModifier::modifyObject(pat::Electron& ele) const {
  auto getVal = [](const pat::Electron& ele, EGEnergySysIndex::Index valIndex) {
    return ele.userFloat(EGEnergySysIndex::name(valIndex));
  };
  //so ele.energy() may be either pre or post corrections, we have no idea
  //so we explicity access the pre and post correction ecal energies
  //we need the pre corrected to properly do the e/p combination
  //we need the post corrected to get et uncertainty
  const float ecalEnergyPostCorr = getVal(ele, EGEnergySysIndex::kEcalPostCorr);
  const float ecalEnergyPreCorr = getVal(ele, EGEnergySysIndex::kEcalPreCorr);
  const float ecalEnergyErrPreCorr = getVal(ele, EGEnergySysIndex::kEcalErrPreCorr);

  //the et cut is in terms of ecal et using the track angle and post corr ecal energy
  const float etUncert = uncertFunc_->val(ele.et() / ele.energy() * ecalEnergyPostCorr);
  const float smear = getVal(ele, EGEnergySysIndex::kSmearValue);
  const float corr = getVal(ele, EGEnergySysIndex::kScaleValue);

  //get the values we have to reset back to
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();
  const auto oldP4 = ele.p4();
  const float oldP4Err = ele.p4Error(reco::GsfElectron::P4_COMBINATION);
  const float oldTrkMomErr = ele.trackMomentumError();

  ele.setCorrectedEcalEnergy(ecalEnergyPreCorr);
  ele.setCorrectedEcalEnergyError(ecalEnergyErrPreCorr);

  const float energyEtUncertUp = calCombinedMom(ele, corr + etUncert, smear).first;
  const float energyEtUncertDn = calCombinedMom(ele, corr - etUncert, smear).first;

  //reset it back to how it was
  ele.setCorrectedEcalEnergy(oldEcalEnergy);
  ele.setCorrectedEcalEnergyError(oldEcalEnergyErr);
  ele.correctMomentum(oldP4, oldTrkMomErr, oldP4Err);

  ele.addUserFloat("energyScaleEtUp", energyEtUncertUp);
  ele.addUserFloat("energyScaleEtDown", energyEtUncertDn);
}

void EGEtScaleSysModifier::modifyObject(pat::Photon& pho) const {
  auto getVal = [](const pat::Photon& pho, EGEnergySysIndex::Index valIndex) {
    return pho.userFloat(EGEnergySysIndex::name(valIndex));
  };
  //so pho.energy() may be either pre or post corrections, we have no idea
  //so we explicity access the pre and post correction ecal energies
  //post corr for the et value for the systematic, pre corr to apply them
  const float ecalEnergyPostCorr = getVal(pho, EGEnergySysIndex::kEcalPostCorr);
  const float ecalEnergyPreCorr = getVal(pho, EGEnergySysIndex::kEcalPreCorr);

  //the et cut is in terms of post corr ecal energy
  const float etUncert = uncertFunc_->val(pho.et() / pho.energy() * ecalEnergyPostCorr);
  const float corr = getVal(pho, EGEnergySysIndex::kScaleValue);

  const float energyEtUncertUp = ecalEnergyPreCorr * (corr + etUncert);
  const float energyEtUncertDn = ecalEnergyPreCorr * (corr - etUncert);

  pho.addUserFloat("energyScaleEtUp", energyEtUncertUp);
  pho.addUserFloat("energyScaleEtDown", energyEtUncertDn);
}

std::pair<float, float> EGEtScaleSysModifier::calCombinedMom(reco::GsfElectron& ele,
                                                             const float scale,
                                                             const float smear) const {
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();
  const auto oldP4 = ele.p4();
  const float oldP4Err = ele.p4Error(reco::GsfElectron::P4_COMBINATION);
  const float oldTrkMomErr = ele.trackMomentumError();

  setEcalEnergy(ele, scale, smear);
  const auto& combinedMomentum = epCombTool_.combine(ele);
  ele.setCorrectedEcalEnergy(oldEcalEnergy);
  ele.setCorrectedEcalEnergyError(oldEcalEnergyErr);
  ele.correctMomentum(oldP4, oldTrkMomErr, oldP4Err);

  return combinedMomentum;
}

void EGEtScaleSysModifier::setEcalEnergy(reco::GsfElectron& ele, const float scale, const float smear) const {
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();
  ele.setCorrectedEcalEnergy(oldEcalEnergy * scale);
  ele.setCorrectedEcalEnergyError(std::hypot(oldEcalEnergyErr * scale, oldEcalEnergy * smear * scale));
}

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGEtScaleSysModifier, "EGEtScaleSysModifier");
