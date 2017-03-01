#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace {
  const edm::EDGetTokenT<edm::ValueMap<float> > empty_token;
  const static edm::InputTag empty_tag("");
}

#include <unordered_map>

class EGFull5x5ShowerShapeModifierFromValueMaps : public ModifyObjectValueBase {
public:
  struct electron_config {
    edm::InputTag electron_src;
    edm::InputTag sigmaEtaEta ;
    edm::InputTag sigmaIetaIeta ;
    edm::InputTag sigmaIphiIphi ;
    edm::InputTag e1x5 ;
    edm::InputTag e2x5Max ;
    edm::InputTag e5x5 ;
    edm::InputTag r9 ;
    edm::InputTag hcalDepth1OverEcal ;     
    edm::InputTag hcalDepth2OverEcal ;
    edm::InputTag hcalDepth1OverEcalBc ; 
    edm::InputTag hcalDepth2OverEcalBc ;
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_sigmaEtaEta ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_sigmaIetaIeta ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_sigmaIphiIphi ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e1x5 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e2x5Max ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e5x5 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_r9 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth1OverEcal ;     
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth2OverEcal ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth1OverEcalBc ; 
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth2OverEcalBc ;
  };

  struct photon_config {
    edm::InputTag photon_src ;
    edm::InputTag sigmaEtaEta ;
    edm::InputTag sigmaIetaIeta ;
    edm::InputTag e1x5 ;
    edm::InputTag e2x5 ;
    edm::InputTag e3x3 ;
    edm::InputTag e5x5 ;
    edm::InputTag maxEnergyXtal ; 
    edm::InputTag hcalDepth1OverEcal ;
    edm::InputTag hcalDepth2OverEcal ;
    edm::InputTag hcalDepth1OverEcalBc;
    edm::InputTag hcalDepth2OverEcalBc;
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_sigmaEtaEta ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_sigmaIetaIeta ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e1x5 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e2x5 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e3x3 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_e5x5 ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_maxEnergyXtal ; 
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth1OverEcal ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth2OverEcal ;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth1OverEcalBc;
    edm::EDGetTokenT<edm::ValueMap<float> > tok_hcalDepth2OverEcalBc;
  };

  EGFull5x5ShowerShapeModifierFromValueMaps(const edm::ParameterSet& conf);
  
  void setEvent(const edm::Event&) override final;
  void setEventContent(const edm::EventSetup&) override final;
  void setConsumes(edm::ConsumesCollector&) override final;
  
  void modifyObject(pat::Electron&) const override final;
  void modifyObject(pat::Photon&) const override final;

private:
  electron_config e_conf;
  photon_config   ph_conf;
  std::unordered_map<unsigned,edm::Ptr<reco::GsfElectron> > eles_by_oop; // indexed by original object ptr
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > ele_vmaps;
  std::unordered_map<unsigned,edm::Ptr<reco::Photon> > phos_by_oop;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > pho_vmaps;
  mutable unsigned ele_idx,pho_idx; // hack here until we figure out why some slimmedPhotons don't have original object ptrs
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGFull5x5ShowerShapeModifierFromValueMaps,
		  "EGFull5x5ShowerShapeModifierFromValueMaps");

EGFull5x5ShowerShapeModifierFromValueMaps::
EGFull5x5ShowerShapeModifierFromValueMaps(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {
  if( conf.exists("electron_config") ) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if( electrons.exists("electronSrc") ) e_conf.electron_src = electrons.getParameter<edm::InputTag>("electronSrc");
    if( electrons.exists("sigmaEtaEta") ) e_conf.sigmaEtaEta = electrons.getParameter<edm::InputTag>("sigmaEtaEta");
    if( electrons.exists("sigmaIetaIeta") ) e_conf.sigmaIetaIeta = electrons.getParameter<edm::InputTag>("sigmaIetaIeta");
    if( electrons.exists("sigmaIphiIphi") ) e_conf.sigmaIphiIphi = electrons.getParameter<edm::InputTag>("sigmaIphiIphi");
    if( electrons.exists("e1x5") ) e_conf.e1x5 = electrons.getParameter<edm::InputTag>("e1x5");
    if( electrons.exists("e2x5Max") ) e_conf.e2x5Max = electrons.getParameter<edm::InputTag>("e2x5Max");
    if( electrons.exists("e5x5") ) e_conf.e5x5 = electrons.getParameter<edm::InputTag>("e5x5");
    if( electrons.exists("r9") ) e_conf.r9 = electrons.getParameter<edm::InputTag>("r9");
    if( electrons.exists("hcalDepth1OverEcal") ) e_conf.hcalDepth1OverEcal = electrons.getParameter<edm::InputTag>("hcalDepth1OverEcal");
    if( electrons.exists("hcalDepth2OverEcal") ) e_conf.hcalDepth2OverEcal = electrons.getParameter<edm::InputTag>("hcalDepth2OverEcal");
    if( electrons.exists("hcalDepth1OverEcalBc") ) e_conf.hcalDepth1OverEcalBc = electrons.getParameter<edm::InputTag>("hcalDepth1OverEcalBc");
    if( electrons.exists("hcalDepth2OverEcalBc") ) e_conf.hcalDepth2OverEcalBc = electrons.getParameter<edm::InputTag>("hcalDepth2OverEcalBc");
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if( photons.exists("photonSrc") ) ph_conf.photon_src = photons.getParameter<edm::InputTag>("photonSrc");
    if( photons.exists("sigmaEtaEta") ) ph_conf.sigmaEtaEta = photons.getParameter<edm::InputTag>("sigmaEtaEta");
    if( photons.exists("sigmaIetaIeta") ) ph_conf.sigmaIetaIeta = photons.getParameter<edm::InputTag>("sigmaIetaIeta");
    if( photons.exists("e1x5") ) ph_conf.e1x5 = photons.getParameter<edm::InputTag>("e1x5");
    if( photons.exists("e2x5") ) ph_conf.e2x5 = photons.getParameter<edm::InputTag>("e2x5");
    if( photons.exists("e3x3") ) ph_conf.e3x3 = photons.getParameter<edm::InputTag>("e3x3");
    if( photons.exists("e5x5") ) ph_conf.e5x5 = photons.getParameter<edm::InputTag>("e5x5");
    if( photons.exists("maxEnergyXtal") ) ph_conf.maxEnergyXtal = photons.getParameter<edm::InputTag>("maxEnergyXtal");
    if( photons.exists("hcalDepth1OverEcal") ) ph_conf.hcalDepth1OverEcal = photons.getParameter<edm::InputTag>("hcalDepth1OverEcal");
    if( photons.exists("hcalDepth2OverEcal") ) ph_conf.hcalDepth2OverEcal = photons.getParameter<edm::InputTag>("hcalDepth2OverEcal");
    if( photons.exists("hcalDepth1OverEcalBc") ) ph_conf.hcalDepth1OverEcalBc = photons.getParameter<edm::InputTag>("hcalDepth1OverEcalBc");
    if( photons.exists("hcalDepth2OverEcalBc") ) ph_conf.hcalDepth2OverEcalBc = photons.getParameter<edm::InputTag>("hcalDepth2OverEcalBc");
  }
  
  ele_idx = pho_idx = 0;
}

namespace {
  inline void get_product(const edm::Event& evt,
                          const edm::EDGetTokenT<edm::ValueMap<float> >& tok,
                          std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float> > >& map) {
    if( !tok.isUninitialized() ) evt.getByToken(tok,map[tok.index()]);
  }
}

void EGFull5x5ShowerShapeModifierFromValueMaps::
setEvent(const edm::Event& evt) {
  eles_by_oop.clear();
  phos_by_oop.clear();  
  ele_vmaps.clear();
  pho_vmaps.clear();

  ele_idx = pho_idx = 0;

  if( !e_conf.tok_electron_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Electron> > eles;
    evt.getByToken(e_conf.tok_electron_src,eles);
    
    for( unsigned i = 0; i < eles->size(); ++i ) {
      edm::Ptr<pat::Electron> ptr = eles->ptrAt(i);
      eles_by_oop[i] = ptr;
    }
  }

  get_product(evt,e_conf.tok_sigmaEtaEta,ele_vmaps);
  get_product(evt,e_conf.tok_sigmaIetaIeta,ele_vmaps);
  get_product(evt,e_conf.tok_sigmaIphiIphi,ele_vmaps);
  get_product(evt,e_conf.tok_e1x5,ele_vmaps);
  get_product(evt,e_conf.tok_e2x5Max,ele_vmaps);
  get_product(evt,e_conf.tok_e5x5,ele_vmaps);
  get_product(evt,e_conf.tok_r9,ele_vmaps);
  get_product(evt,e_conf.tok_hcalDepth1OverEcal,ele_vmaps);
  get_product(evt,e_conf.tok_hcalDepth2OverEcal,ele_vmaps);
  get_product(evt,e_conf.tok_hcalDepth1OverEcalBc,ele_vmaps);
  get_product(evt,e_conf.tok_hcalDepth2OverEcalBc,ele_vmaps);

  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);

    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[i] = ptr;
    }
  }

  get_product(evt,ph_conf.tok_sigmaEtaEta,pho_vmaps);
  get_product(evt,ph_conf.tok_sigmaIetaIeta,pho_vmaps);
  get_product(evt,ph_conf.tok_e1x5,pho_vmaps);
  get_product(evt,ph_conf.tok_e2x5,pho_vmaps);
  get_product(evt,ph_conf.tok_e3x3,pho_vmaps);
  get_product(evt,ph_conf.tok_e5x5,pho_vmaps);
  get_product(evt,ph_conf.tok_maxEnergyXtal,pho_vmaps);
  get_product(evt,ph_conf.tok_hcalDepth1OverEcal,pho_vmaps);
  get_product(evt,ph_conf.tok_hcalDepth2OverEcal,pho_vmaps);
  get_product(evt,ph_conf.tok_hcalDepth1OverEcalBc,pho_vmaps);
  get_product(evt,ph_conf.tok_hcalDepth2OverEcalBc,pho_vmaps);

}

void EGFull5x5ShowerShapeModifierFromValueMaps::
setEventContent(const edm::EventSetup& evs) {
}

namespace {
  template<typename T, typename U, typename V>
  inline void make_consumes(T& tag,U& tok,V& sume) { if( !(empty_tag == tag) ) tok = sume.template consumes<edm::ValueMap<float> >(tag); }
}

void EGFull5x5ShowerShapeModifierFromValueMaps::
setConsumes(edm::ConsumesCollector& sumes) {
  //setup electrons
  if( !(empty_tag == e_conf.electron_src) ) e_conf.tok_electron_src = sumes.consumes<edm::View<pat::Electron> >(e_conf.electron_src);  
  make_consumes(e_conf.sigmaEtaEta,e_conf.tok_sigmaEtaEta,sumes);
  make_consumes(e_conf.sigmaIetaIeta,e_conf.tok_sigmaIetaIeta,sumes);
  make_consumes(e_conf.sigmaIphiIphi,e_conf.tok_sigmaIphiIphi,sumes);
  make_consumes(e_conf.e1x5,e_conf.tok_e1x5,sumes);
  make_consumes(e_conf.e2x5Max,e_conf.tok_e2x5Max,sumes);
  make_consumes(e_conf.e5x5,e_conf.tok_e5x5,sumes);
  make_consumes(e_conf.r9,e_conf.tok_r9,sumes);
  make_consumes(e_conf.hcalDepth1OverEcal,e_conf.tok_hcalDepth1OverEcal,sumes);
  make_consumes(e_conf.hcalDepth2OverEcal,e_conf.tok_hcalDepth2OverEcal,sumes);
  make_consumes(e_conf.hcalDepth1OverEcalBc,e_conf.tok_hcalDepth1OverEcalBc,sumes);
  make_consumes(e_conf.hcalDepth2OverEcalBc,e_conf.tok_hcalDepth2OverEcalBc,sumes);    

  // setup photons 
  if( !(empty_tag == ph_conf.photon_src) ) ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);
  make_consumes(ph_conf.sigmaEtaEta,ph_conf.tok_sigmaEtaEta,sumes);
  make_consumes(ph_conf.sigmaIetaIeta,ph_conf.tok_sigmaIetaIeta,sumes);
  make_consumes(ph_conf.e1x5,ph_conf.tok_e1x5,sumes);
  make_consumes(ph_conf.e2x5,ph_conf.tok_e2x5,sumes);
  make_consumes(ph_conf.e3x3,ph_conf.tok_e3x3,sumes);
  make_consumes(ph_conf.e5x5,ph_conf.tok_e5x5,sumes);
  make_consumes(ph_conf.maxEnergyXtal,ph_conf.tok_maxEnergyXtal,sumes);
  make_consumes(ph_conf.hcalDepth1OverEcal,ph_conf.tok_hcalDepth1OverEcal,sumes);
  make_consumes(ph_conf.hcalDepth2OverEcal,ph_conf.tok_hcalDepth2OverEcal,sumes);
  make_consumes(ph_conf.hcalDepth1OverEcalBc,ph_conf.tok_hcalDepth1OverEcalBc,sumes);
  make_consumes(ph_conf.hcalDepth2OverEcalBc,ph_conf.tok_hcalDepth2OverEcalBc,sumes);   
}

namespace {
  template<typename T, typename U, typename V>
  inline void assignValue(const T& ptr, const U& tok, const V& map, float& value) {
    if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
  }
}

void EGFull5x5ShowerShapeModifierFromValueMaps::
modifyObject(pat::Electron& ele) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference    
  edm::Ptr<reco::Candidate> ptr(ele.originalObjectRef());
  if( !e_conf.tok_electron_src.isUninitialized() ) {
    auto key = eles_by_oop.find(ele_idx);
    if( key != eles_by_oop.end() ) {
      ptr = key->second;
    } else {
      throw cms::Exception("BadElectronKey")
        << "Original object pointer with key = " << ele.originalObjectRef().key() << " not found in cache!";
    }
  }
  //now we go through and modify the objects using the valuemaps we read in
  auto full5x5 = ele.full5x5_showerShape();
  assignValue(ptr,e_conf.tok_sigmaEtaEta,ele_vmaps,full5x5.sigmaEtaEta);
  assignValue(ptr,e_conf.tok_sigmaIetaIeta,ele_vmaps,full5x5.sigmaIetaIeta);
  assignValue(ptr,e_conf.tok_sigmaIphiIphi,ele_vmaps,full5x5.sigmaIphiIphi);
  assignValue(ptr,e_conf.tok_e1x5,ele_vmaps,full5x5.e1x5);
  assignValue(ptr,e_conf.tok_e2x5Max,ele_vmaps,full5x5.e2x5Max);
  assignValue(ptr,e_conf.tok_e5x5,ele_vmaps,full5x5.e5x5);
  assignValue(ptr,e_conf.tok_r9,ele_vmaps,full5x5.r9);
  assignValue(ptr,e_conf.tok_hcalDepth1OverEcal,ele_vmaps,full5x5.hcalDepth1OverEcal);
  assignValue(ptr,e_conf.tok_hcalDepth2OverEcal,ele_vmaps,full5x5.hcalDepth2OverEcal);
  assignValue(ptr,e_conf.tok_hcalDepth1OverEcalBc,ele_vmaps,full5x5.hcalDepth1OverEcalBc);
  assignValue(ptr,e_conf.tok_hcalDepth2OverEcalBc,ele_vmaps,full5x5.hcalDepth2OverEcalBc);

  ele.full5x5_setShowerShape(full5x5);
  ++ele_idx;
}


void EGFull5x5ShowerShapeModifierFromValueMaps::
modifyObject(pat::Photon& pho) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());
  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    auto key = phos_by_oop.find(pho_idx);
    if( key != phos_by_oop.end() ) {
      ptr = key->second;
    } else {
      throw cms::Exception("BadPhotonKey")
        << "Original object pointer with key = " << pho.originalObjectRef().key() << " not found in cache!";
    }
  }
  //now we go through and modify the objects using the valuemaps we read in
  auto full5x5 = pho.full5x5_showerShapeVariables();
  assignValue(ptr,ph_conf.tok_sigmaEtaEta,pho_vmaps,full5x5.sigmaEtaEta);
  assignValue(ptr,ph_conf.tok_sigmaIetaIeta,pho_vmaps,full5x5.sigmaIetaIeta);
  assignValue(ptr,ph_conf.tok_e1x5,pho_vmaps,full5x5.e1x5);
  assignValue(ptr,ph_conf.tok_e2x5,pho_vmaps,full5x5.e2x5);
  assignValue(ptr,ph_conf.tok_e3x3,pho_vmaps,full5x5.e3x3);
  assignValue(ptr,ph_conf.tok_e5x5,pho_vmaps,full5x5.e5x5);
  assignValue(ptr,ph_conf.tok_maxEnergyXtal,pho_vmaps,full5x5.maxEnergyXtal);
  assignValue(ptr,ph_conf.tok_hcalDepth1OverEcal,pho_vmaps,full5x5.hcalDepth1OverEcal);
  assignValue(ptr,ph_conf.tok_hcalDepth2OverEcal,pho_vmaps,full5x5.hcalDepth2OverEcal);
  assignValue(ptr,ph_conf.tok_hcalDepth1OverEcalBc,pho_vmaps,full5x5.hcalDepth1OverEcalBc);
  assignValue(ptr,ph_conf.tok_hcalDepth2OverEcalBc,pho_vmaps,full5x5.hcalDepth2OverEcalBc);

  pho.full5x5_setShowerShapeVariables(full5x5);
  ++pho_idx;
}
