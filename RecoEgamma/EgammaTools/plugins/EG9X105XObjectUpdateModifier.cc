#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"

#include <vdt/vdtMath.h>

//this modifier fills variables where not present in CMSSW_92X to CMSSW_105X
//use case is when reading older new samples in newer releases, aka legacy
//note we suffer from the problem of needing to use the electrons/photons the valuemaps
//are keyed to (something that is thankfully going away in 11X!) so we have to take
//those collections in and figure out which ele/pho matches to them
class EG9X105XObjectUpdateModifier : public ModifyObjectValueBase {
public:
  template <typename T>
  class TokenHandlePair {
  public:
    TokenHandlePair(const edm::ParameterSet& conf, const std::string& name, edm::ConsumesCollector& cc)
        : token_(cc.consumes(conf.getParameter<edm::InputTag>(name))) {}
    void setHandle(const edm::Event& iEvent) { handle_ = iEvent.getHandle(token_); }
    const edm::Handle<T>& handle() const { return handle_; }

  private:
    edm::EDGetTokenT<T> token_;
    edm::Handle<T> handle_;
  };

  EG9X105XObjectUpdateModifier(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);
  ~EG9X105XObjectUpdateModifier() override {}

  void setEvent(const edm::Event&) final;

  void modifyObject(reco::GsfElectron& ele) const final;
  void modifyObject(reco::Photon& pho) const final;

  void modifyObject(pat::Electron& ele) const final { return modifyObject(static_cast<reco::GsfElectron&>(ele)); }
  void modifyObject(pat::Photon& pho) const final { return modifyObject(static_cast<reco::Photon&>(pho)); }

private:
  template <typename ObjType>
  static edm::Ptr<ObjType> getPtrForValueMap(const ObjType& obj,
                                             const edm::Handle<edm::View<ObjType> >& objsVMIsKeyedTo);

  TokenHandlePair<edm::View<reco::GsfElectron> > eleCollVMsAreKeyedTo_;
  TokenHandlePair<edm::View<reco::Photon> > phoCollVMsAreKeyedTo_;

  TokenHandlePair<reco::ConversionCollection> conversions_;
  TokenHandlePair<reco::BeamSpot> beamspot_;
  TokenHandlePair<EcalRecHitCollection> ecalRecHitsEB_;
  TokenHandlePair<EcalRecHitCollection> ecalRecHitsEE_;

  TokenHandlePair<edm::ValueMap<float> > eleTrkIso_;
  TokenHandlePair<edm::ValueMap<float> > eleTrkIso04_;
  TokenHandlePair<edm::ValueMap<float> > phoPhotonIso_;
  TokenHandlePair<edm::ValueMap<float> > phoNeutralHadIso_;
  TokenHandlePair<edm::ValueMap<float> > phoChargedHadIso_;
  TokenHandlePair<edm::ValueMap<float> > phoChargedHadWorstVtxIso_;
  TokenHandlePair<edm::ValueMap<float> > phoChargedHadWorstVtxConeVetoIso_;
  TokenHandlePair<edm::ValueMap<float> > phoChargedHadPFPVIso_;
  //there is a bug which GsfTracks are now allowed to be a match for conversions
  //due to improper linking of references in the miniAOD since 94X
  //this allows us to emulate it or not
  //note: even if this enabled, it will do nothing on miniAOD produced with 94X, 102X
  //till upto whenever this is fixed (11X?) as the GsfTrack references point to a different
  //collection to the conversion track references
  bool allowGsfTrkMatchForConvs_;
  //this allows us to update the charged hadron PF PV isolation
  //chargedHadPFPVIso is filled in iorules but when running on miniAOD, the value used in IDs
  //is remade on the miniAOD packedcandidates which differs due to rounding
  //its still the same variable but can have differences hence inorder to allow IDs calculated on miniAOD
  //on the same file to be exactly reproduced, this option is set true
  bool updateChargedHadPFPVIso_;
};

EG9X105XObjectUpdateModifier::EG9X105XObjectUpdateModifier(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf),
      eleCollVMsAreKeyedTo_(conf, "eleCollVMsAreKeyedTo", cc),
      phoCollVMsAreKeyedTo_(conf, "phoCollVMsAreKeyedTo", cc),
      conversions_(conf, "conversions", cc),
      beamspot_(conf, "beamspot", cc),
      ecalRecHitsEB_(conf, "ecalRecHitsEB", cc),
      ecalRecHitsEE_(conf, "ecalRecHitsEE", cc),
      eleTrkIso_(conf, "eleTrkIso", cc),
      eleTrkIso04_(conf, "eleTrkIso04", cc),
      phoPhotonIso_(conf, "phoPhotonIso", cc),
      phoNeutralHadIso_(conf, "phoNeutralHadIso", cc),
      phoChargedHadIso_(conf, "phoChargedHadIso", cc),
      phoChargedHadWorstVtxIso_(conf, "phoChargedHadWorstVtxIso", cc),
      phoChargedHadWorstVtxConeVetoIso_(conf, "phoChargedHadWorstVtxConeVetoIso", cc),
      phoChargedHadPFPVIso_(conf, "phoChargedHadPFPVIso", cc),
      allowGsfTrkMatchForConvs_(conf.getParameter<bool>("allowGsfTrackForConvs")),
      updateChargedHadPFPVIso_(conf.getParameter<bool>("updateChargedHadPFPVIso")) {}

void EG9X105XObjectUpdateModifier::setEvent(const edm::Event& iEvent) {
  eleCollVMsAreKeyedTo_.setHandle(iEvent);
  phoCollVMsAreKeyedTo_.setHandle(iEvent);
  conversions_.setHandle(iEvent);
  beamspot_.setHandle(iEvent);
  ecalRecHitsEB_.setHandle(iEvent);
  ecalRecHitsEE_.setHandle(iEvent);
  eleTrkIso_.setHandle(iEvent);
  eleTrkIso04_.setHandle(iEvent);
  phoPhotonIso_.setHandle(iEvent);
  phoNeutralHadIso_.setHandle(iEvent);
  phoChargedHadIso_.setHandle(iEvent);
  phoChargedHadWorstVtxIso_.setHandle(iEvent);
  phoChargedHadWorstVtxConeVetoIso_.setHandle(iEvent);
  if (updateChargedHadPFPVIso_)
    phoChargedHadPFPVIso_.setHandle(iEvent);
}

void EG9X105XObjectUpdateModifier::modifyObject(reco::GsfElectron& ele) const {
  edm::Ptr<reco::GsfElectron> ptrForVM = getPtrForValueMap(ele, eleCollVMsAreKeyedTo_.handle());
  if (ptrForVM.isNull()) {
    throw cms::Exception("LogicError")
        << " in EG9X105ObjectUpdateModifier, line " << __LINE__ << " electron " << ele.et() << " " << ele.eta() << " "
        << ele.superCluster()->seed()->seed().rawId()
        << " failed to match to the electrons the key map was keyed to, check the map collection is correct";
  }
  reco::GsfElectron::ConversionRejection convRejVars = ele.conversionRejectionVariables();
  if (allowGsfTrkMatchForConvs_) {
    convRejVars.vtxFitProb = ConversionTools::getVtxFitProb(
        ConversionTools::matchedConversion(ele.core(), *conversions_.handle(), beamspot_.handle()->position()));
  } else {
    //its rather important to use the core function here to get the org trk ref
    convRejVars.vtxFitProb = ConversionTools::getVtxFitProb(ConversionTools::matchedConversion(
        ele.core()->ctfTrack(), *conversions_.handle(), beamspot_.handle()->position(), 2.0, 1e-6, 0));
  }
  ele.setConversionRejectionVariables(convRejVars);

  reco::GsfElectron::IsolationVariables isolVars03 = ele.dr03IsolationVariables();
  isolVars03.tkSumPtHEEP = (*eleTrkIso_.handle())[ptrForVM];
  ele.setDr03Isolation(isolVars03);
  reco::GsfElectron::IsolationVariables isolVars04 = ele.dr04IsolationVariables();
  isolVars04.tkSumPtHEEP = (*eleTrkIso04_.handle())[ptrForVM];
  ele.setDr04Isolation(isolVars04);
}

void EG9X105XObjectUpdateModifier::modifyObject(reco::Photon& pho) const {
  edm::Ptr<reco::Photon> ptrForVM = getPtrForValueMap(pho, phoCollVMsAreKeyedTo_.handle());
  if (ptrForVM.isNull()) {
    throw cms::Exception("LogicError")
        << " in EG9X105ObjectUpdateModifier, line " << __LINE__ << " photon " << pho.et() << " " << pho.eta() << " "
        << pho.superCluster()->seed()->seed().rawId()
        << " failed to match to the photons the key map was keyed to, check the map collection is correct";
  }

  reco::Photon::PflowIsolationVariables pfIso = pho.getPflowIsolationVariables();
  pfIso.photonIso = (*phoPhotonIso_.handle())[ptrForVM];
  pfIso.neutralHadronIso = (*phoNeutralHadIso_.handle())[ptrForVM];
  pfIso.chargedHadronIso = (*phoChargedHadIso_.handle())[ptrForVM];
  pfIso.chargedHadronWorstVtxIso = (*phoChargedHadWorstVtxIso_.handle())[ptrForVM];
  pfIso.chargedHadronWorstVtxGeomVetoIso = (*phoChargedHadWorstVtxConeVetoIso_.handle())[ptrForVM];
  if (updateChargedHadPFPVIso_) {
    pfIso.chargedHadronPFPVIso = (*phoChargedHadPFPVIso_.handle())[ptrForVM];
  }
  pho.setPflowIsolationVariables(pfIso);

  reco::Photon::ShowerShape fracSS = pho.showerShapeVariables();
  reco::Photon::ShowerShape fullSS = pho.full5x5_showerShapeVariables();

  const reco::CaloClusterPtr seedClus = pho.superCluster()->seed();
  const bool isEB = seedClus->seed().subdetId() == EcalBarrel;
  const auto& recHits = isEB ? *ecalRecHitsEB_.handle() : *ecalRecHitsEE_.handle();
  Cluster2ndMoments clus2ndMomFrac = EcalClusterTools::cluster2ndMoments(*seedClus, recHits);
  Cluster2ndMoments clus2ndMomFull = noZS::EcalClusterTools::cluster2ndMoments(*seedClus, recHits);
  fracSS.smMajor = clus2ndMomFrac.sMaj;
  fracSS.smMinor = clus2ndMomFrac.sMin;
  fracSS.smAlpha = clus2ndMomFrac.alpha;
  fullSS.smMajor = clus2ndMomFull.sMaj;
  fullSS.smMinor = clus2ndMomFull.sMin;
  fullSS.smAlpha = clus2ndMomFull.alpha;
  pho.setShowerShapeVariables(fracSS);
  pho.full5x5_setShowerShapeVariables(fullSS);
}

template <typename ObjType>
edm::Ptr<ObjType> EG9X105XObjectUpdateModifier::getPtrForValueMap(
    const ObjType& obj, const edm::Handle<edm::View<ObjType> >& objsVMIsKeyedTo) {
  for (auto& objVMPtr : objsVMIsKeyedTo->ptrs()) {
    if (obj.superCluster()->seed()->seed() == objVMPtr->superCluster()->seed()->seed())
      return objVMPtr;
  }
  return edm::Ptr<ObjType>(objsVMIsKeyedTo.id());  //return null ptr if not found
}

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EG9X105XObjectUpdateModifier, "EG9X105XObjectUpdateModifier");
