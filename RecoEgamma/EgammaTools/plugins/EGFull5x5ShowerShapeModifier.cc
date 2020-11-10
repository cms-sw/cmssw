#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace {
  const edm::InputTag empty_tag("");
  template <typename T, typename U, typename V>
  inline void make_consumes(const T& tag, edm::EDGetTokenT<U>& tok, V& sume) {
    if (!(empty_tag == tag))
      tok = sume.template consumes(tag);
  }
}  // namespace

#include <unordered_map>

class EGFull5x5ShowerShapeModifierFromValueMaps : public ModifyObjectValueBase {
public:
  struct electron_config {
    edm::EDGetTokenT<edm::View<pat::Electron>> tok_electron_src;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_sigmaEtaEta;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_sigmaIetaIeta;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_sigmaIphiIphi;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e1x5;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e2x5Max;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e5x5;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_r9;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth1OverEcal;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth2OverEcal;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth1OverEcalBc;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth2OverEcalBc;
  };

  struct photon_config {
    edm::EDGetTokenT<edm::View<pat::Photon>> tok_photon_src;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_sigmaEtaEta;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_sigmaIetaIeta;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e1x5;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e2x5;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e3x3;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_e5x5;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_maxEnergyXtal;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth1OverEcal;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth2OverEcal;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth1OverEcalBc;
    edm::EDGetTokenT<edm::ValueMap<float>> tok_hcalDepth2OverEcalBc;
  };

  EGFull5x5ShowerShapeModifierFromValueMaps(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);

  void setEvent(const edm::Event&) final;

  void modifyObject(pat::Electron&) const final;
  void modifyObject(pat::Photon&) const final;

private:
  electron_config e_conf;
  photon_config ph_conf;
  std::vector<edm::Ptr<reco::GsfElectron>> eles_by_oop;  // indexed by original object ptr
  std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float>>> ele_vmaps;
  std::vector<edm::Ptr<reco::Photon>> phos_by_oop;
  std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float>>> pho_vmaps;
  mutable unsigned ele_idx,
      pho_idx;  // hack here until we figure out why some slimmedPhotons don't have original object ptrs
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
                  EGFull5x5ShowerShapeModifierFromValueMaps,
                  "EGFull5x5ShowerShapeModifierFromValueMaps");

EGFull5x5ShowerShapeModifierFromValueMaps::EGFull5x5ShowerShapeModifierFromValueMaps(const edm::ParameterSet& conf,
                                                                                     edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf) {
  if (conf.exists("electron_config")) {
    auto const& electrons = conf.getParameterSet("electron_config");
    if (electrons.exists("electronSrc"))
      make_consumes(electrons.getParameter<edm::InputTag>("electronSrc"), e_conf.tok_electron_src, cc);
    if (electrons.exists("sigmaEtaEta"))
      make_consumes(electrons.getParameter<edm::InputTag>("sigmaEtaEta"), e_conf.tok_sigmaEtaEta, cc);
    if (electrons.exists("sigmaIetaIeta"))
      make_consumes(electrons.getParameter<edm::InputTag>("sigmaIetaIeta"), e_conf.tok_sigmaIetaIeta, cc);
    if (electrons.exists("sigmaIphiIphi"))
      make_consumes(electrons.getParameter<edm::InputTag>("sigmaIphiIphi"), e_conf.tok_sigmaIphiIphi, cc);
    if (electrons.exists("e1x5"))
      make_consumes(electrons.getParameter<edm::InputTag>("e1x5"), e_conf.tok_e1x5, cc);
    if (electrons.exists("e2x5Max"))
      make_consumes(electrons.getParameter<edm::InputTag>("e2x5Max"), e_conf.tok_e2x5Max, cc);
    if (electrons.exists("e5x5"))
      make_consumes(electrons.getParameter<edm::InputTag>("e5x5"), e_conf.tok_e5x5, cc);
    if (electrons.exists("r9"))
      make_consumes(electrons.getParameter<edm::InputTag>("r9"), e_conf.tok_r9, cc);
    if (electrons.exists("hcalDepth1OverEcal"))
      make_consumes(electrons.getParameter<edm::InputTag>("hcalDepth1OverEcal"), e_conf.tok_hcalDepth1OverEcal, cc);
    if (electrons.exists("hcalDepth2OverEcal"))
      make_consumes(electrons.getParameter<edm::InputTag>("hcalDepth2OverEcal"), e_conf.tok_hcalDepth2OverEcal, cc);
    if (electrons.exists("hcalDepth1OverEcalBc"))
      make_consumes(electrons.getParameter<edm::InputTag>("hcalDepth1OverEcalBc"), e_conf.tok_hcalDepth1OverEcalBc, cc);
    if (electrons.exists("hcalDepth2OverEcalBc"))
      make_consumes(electrons.getParameter<edm::InputTag>("hcalDepth2OverEcalBc"), e_conf.tok_hcalDepth2OverEcalBc, cc);
  }
  if (conf.exists("photon_config")) {
    auto const& photons = conf.getParameterSet("photon_config");
    if (photons.exists("photonSrc"))
      make_consumes(photons.getParameter<edm::InputTag>("photonSrc"), ph_conf.tok_photon_src, cc);
    if (photons.exists("sigmaEtaEta"))
      make_consumes(photons.getParameter<edm::InputTag>("sigmaEtaEta"), ph_conf.tok_sigmaEtaEta, cc);
    if (photons.exists("sigmaIetaIeta"))
      make_consumes(photons.getParameter<edm::InputTag>("sigmaIetaIeta"), ph_conf.tok_sigmaIetaIeta, cc);
    if (photons.exists("e1x5"))
      make_consumes(photons.getParameter<edm::InputTag>("e1x5"), ph_conf.tok_e1x5, cc);
    if (photons.exists("e2x5"))
      make_consumes(photons.getParameter<edm::InputTag>("e2x5"), ph_conf.tok_e2x5, cc);
    if (photons.exists("e3x3"))
      make_consumes(photons.getParameter<edm::InputTag>("e3x3"), ph_conf.tok_e3x3, cc);
    if (photons.exists("e5x5"))
      make_consumes(photons.getParameter<edm::InputTag>("e5x5"), ph_conf.tok_e5x5, cc);
    if (photons.exists("maxEnergyXtal"))
      make_consumes(photons.getParameter<edm::InputTag>("maxEnergyXtal"), ph_conf.tok_maxEnergyXtal, cc);
    if (photons.exists("hcalDepth1OverEcal"))
      make_consumes(photons.getParameter<edm::InputTag>("hcalDepth1OverEcal"), ph_conf.tok_hcalDepth1OverEcal, cc);
    if (photons.exists("hcalDepth2OverEcal"))
      make_consumes(photons.getParameter<edm::InputTag>("hcalDepth2OverEcal"), ph_conf.tok_hcalDepth2OverEcal, cc);
    if (photons.exists("hcalDepth1OverEcalBc"))
      make_consumes(photons.getParameter<edm::InputTag>("hcalDepth1OverEcalBc"), ph_conf.tok_hcalDepth1OverEcalBc, cc);
    if (photons.exists("hcalDepth2OverEcalBc"))
      make_consumes(photons.getParameter<edm::InputTag>("hcalDepth2OverEcalBc"), ph_conf.tok_hcalDepth2OverEcalBc, cc);
  }

  ele_idx = pho_idx = 0;
}

namespace {
  inline void get_product(const edm::Event& evt,
                          const edm::EDGetTokenT<edm::ValueMap<float>>& tok,
                          std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float>>>& map) {
    if (!tok.isUninitialized())
      map[tok.index()] = evt.getHandle(tok);
  }
}  // namespace

void EGFull5x5ShowerShapeModifierFromValueMaps::setEvent(const edm::Event& evt) {
  eles_by_oop.clear();
  phos_by_oop.clear();
  ele_vmaps.clear();
  pho_vmaps.clear();

  ele_idx = pho_idx = 0;

  if (!e_conf.tok_electron_src.isUninitialized()) {
    auto eles = evt.getHandle(e_conf.tok_electron_src);

    eles_by_oop.resize(eles->size());
    std::copy(eles->ptrs().begin(), eles->ptrs().end(), eles_by_oop.begin());
  }

  get_product(evt, e_conf.tok_sigmaEtaEta, ele_vmaps);
  get_product(evt, e_conf.tok_sigmaIetaIeta, ele_vmaps);
  get_product(evt, e_conf.tok_sigmaIphiIphi, ele_vmaps);
  get_product(evt, e_conf.tok_e1x5, ele_vmaps);
  get_product(evt, e_conf.tok_e2x5Max, ele_vmaps);
  get_product(evt, e_conf.tok_e5x5, ele_vmaps);
  get_product(evt, e_conf.tok_r9, ele_vmaps);
  get_product(evt, e_conf.tok_hcalDepth1OverEcal, ele_vmaps);
  get_product(evt, e_conf.tok_hcalDepth2OverEcal, ele_vmaps);
  get_product(evt, e_conf.tok_hcalDepth1OverEcalBc, ele_vmaps);
  get_product(evt, e_conf.tok_hcalDepth2OverEcalBc, ele_vmaps);

  if (!ph_conf.tok_photon_src.isUninitialized()) {
    auto phos = evt.getHandle(ph_conf.tok_photon_src);

    phos_by_oop.resize(phos->size());
    std::copy(phos->ptrs().begin(), phos->ptrs().end(), phos_by_oop.begin());
  }

  get_product(evt, ph_conf.tok_sigmaEtaEta, pho_vmaps);
  get_product(evt, ph_conf.tok_sigmaIetaIeta, pho_vmaps);
  get_product(evt, ph_conf.tok_e1x5, pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5, pho_vmaps);
  get_product(evt, ph_conf.tok_e3x3, pho_vmaps);
  get_product(evt, ph_conf.tok_e5x5, pho_vmaps);
  get_product(evt, ph_conf.tok_maxEnergyXtal, pho_vmaps);
  get_product(evt, ph_conf.tok_hcalDepth1OverEcal, pho_vmaps);
  get_product(evt, ph_conf.tok_hcalDepth2OverEcal, pho_vmaps);
  get_product(evt, ph_conf.tok_hcalDepth1OverEcalBc, pho_vmaps);
  get_product(evt, ph_conf.tok_hcalDepth2OverEcalBc, pho_vmaps);
}

namespace {
  template <typename T, typename U, typename V>
  inline void assignValue(const T& ptr, const U& tok, const V& map, float& value) {
    if (!tok.isUninitialized())
      value = map.find(tok.index())->second->get(ptr.id(), ptr.key());
  }
}  // namespace

void EGFull5x5ShowerShapeModifierFromValueMaps::modifyObject(pat::Electron& ele) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(ele.originalObjectRef());

  // The calls to this function should be matched to the order of the electrons
  // in eles_by_oop. In case it is called too many times, it will throw thanks
  // to the use of std::vector<T>::at().
  if (!e_conf.tok_electron_src.isUninitialized())
    ptr = eles_by_oop.at(ele_idx);

  //now we go through and modify the objects using the valuemaps we read in
  auto full5x5 = ele.full5x5_showerShape();
  assignValue(ptr, e_conf.tok_sigmaEtaEta, ele_vmaps, full5x5.sigmaEtaEta);
  assignValue(ptr, e_conf.tok_sigmaIetaIeta, ele_vmaps, full5x5.sigmaIetaIeta);
  assignValue(ptr, e_conf.tok_sigmaIphiIphi, ele_vmaps, full5x5.sigmaIphiIphi);
  assignValue(ptr, e_conf.tok_e1x5, ele_vmaps, full5x5.e1x5);
  assignValue(ptr, e_conf.tok_e2x5Max, ele_vmaps, full5x5.e2x5Max);
  assignValue(ptr, e_conf.tok_e5x5, ele_vmaps, full5x5.e5x5);
  assignValue(ptr, e_conf.tok_r9, ele_vmaps, full5x5.r9);
  assignValue(ptr, e_conf.tok_hcalDepth1OverEcal, ele_vmaps, full5x5.hcalDepth1OverEcal);
  assignValue(ptr, e_conf.tok_hcalDepth2OverEcal, ele_vmaps, full5x5.hcalDepth2OverEcal);
  assignValue(ptr, e_conf.tok_hcalDepth1OverEcalBc, ele_vmaps, full5x5.hcalDepth1OverEcalBc);
  assignValue(ptr, e_conf.tok_hcalDepth2OverEcalBc, ele_vmaps, full5x5.hcalDepth2OverEcalBc);

  ele.full5x5_setShowerShape(full5x5);
  ++ele_idx;
}

void EGFull5x5ShowerShapeModifierFromValueMaps::modifyObject(pat::Photon& pho) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());

  // The calls to this function should be matched to the order of the electrons
  // in eles_by_oop. In case it is called too many times, it will throw thanks
  // to the use of std::vector<T>::at().
  if (!ph_conf.tok_photon_src.isUninitialized())
    ptr = phos_by_oop.at(pho_idx);

  //now we go through and modify the objects using the valuemaps we read in
  auto full5x5 = pho.full5x5_showerShapeVariables();
  assignValue(ptr, ph_conf.tok_sigmaEtaEta, pho_vmaps, full5x5.sigmaEtaEta);
  assignValue(ptr, ph_conf.tok_sigmaIetaIeta, pho_vmaps, full5x5.sigmaIetaIeta);
  assignValue(ptr, ph_conf.tok_e1x5, pho_vmaps, full5x5.e1x5);
  assignValue(ptr, ph_conf.tok_e2x5, pho_vmaps, full5x5.e2x5);
  assignValue(ptr, ph_conf.tok_e3x3, pho_vmaps, full5x5.e3x3);
  assignValue(ptr, ph_conf.tok_e5x5, pho_vmaps, full5x5.e5x5);
  assignValue(ptr, ph_conf.tok_maxEnergyXtal, pho_vmaps, full5x5.maxEnergyXtal);
  assignValue(ptr, ph_conf.tok_hcalDepth1OverEcal, pho_vmaps, full5x5.hcalDepth1OverEcal);
  assignValue(ptr, ph_conf.tok_hcalDepth2OverEcal, pho_vmaps, full5x5.hcalDepth2OverEcal);
  assignValue(ptr, ph_conf.tok_hcalDepth1OverEcalBc, pho_vmaps, full5x5.hcalDepth1OverEcalBc);
  assignValue(ptr, ph_conf.tok_hcalDepth2OverEcalBc, pho_vmaps, full5x5.hcalDepth2OverEcalBc);

  pho.full5x5_setShowerShapeVariables(full5x5);
  ++pho_idx;
}
