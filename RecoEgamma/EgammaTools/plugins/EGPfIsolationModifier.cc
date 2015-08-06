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

class EGPfIsolationModifierFromValueMaps : public ModifyObjectValueBase {
public:
  struct electron_config {
    edm::InputTag electron_src;
    edm::InputTag sumChargedHadronPt;
    edm::InputTag sumNeutralHadronEt;
    edm::InputTag sumPhotonEt;
    edm::InputTag sumChargedParticlePt;
    edm::InputTag sumNeutralHadronEtHighThreshold;
    edm::InputTag sumPhotonEtHighThreshold;
    edm::InputTag sumPUPt;
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumChargedHadronPt;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumNeutralHadronEt;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumPhotonEt;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumChargedParticlePt;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumNeutralHadronEtHighThreshold;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumPhotonEtHighThreshold;
    edm::EDGetTokenT<edm::ValueMap<float> >     tok_sumPUPt;
  };

  struct photon_config {
    edm::InputTag photon_src ;
    edm::InputTag chargedHadronIso; 
    edm::InputTag chargedHadronIsoWrongVtx; 
    edm::InputTag neutralHadronIso; 
    edm::InputTag photonIso ;       
    edm::InputTag modFrixione ;      
    edm::InputTag sumChargedParticlePt; 
    edm::InputTag sumNeutralHadronEtHighThreshold;  
    edm::InputTag sumPhotonEtHighThreshold;  
    edm::InputTag sumPUPt;  
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_chargedHadronIso; 
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_chargedHadronIsoWrongVtx; 
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_neutralHadronIso; 
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_photonIso ;       
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_modFrixione ;      
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_sumChargedParticlePt; 
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_sumNeutralHadronEtHighThreshold;  
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_sumPhotonEtHighThreshold;  
    edm::EDGetTokenT<edm::ValueMap<float> >   tok_sumPUPt;  
  };

  EGPfIsolationModifierFromValueMaps(const edm::ParameterSet& conf);
  
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
		  EGPfIsolationModifierFromValueMaps,
		  "EGPfIsolationModifierFromValueMaps");

EGPfIsolationModifierFromValueMaps::
EGPfIsolationModifierFromValueMaps(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {
  if( conf.exists("electron_config") ) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if( electrons.exists("electronSrc") ) e_conf.electron_src = electrons.getParameter<edm::InputTag>("electronSrc");
    if( electrons.exists("sumChargedHadronPt") ) e_conf.sumChargedHadronPt = electrons.getParameter<edm::InputTag>("sumChargedHadronPt");
    if( electrons.exists("sumNeutralHadronEt") ) e_conf.sumNeutralHadronEt = electrons.getParameter<edm::InputTag>("sumNeutralHadronPt");
    if( electrons.exists("sumPhotonEt") ) e_conf.sumPhotonEt = electrons.getParameter<edm::InputTag>("sumPhotonEt");
    if( electrons.exists("sumChargedParticlePt") ) e_conf.sumChargedParticlePt = electrons.getParameter<edm::InputTag>("sumChargedParticlePt");
    if( electrons.exists("sumNeutralHadronEtHighThreshold") ) e_conf.sumNeutralHadronEtHighThreshold = electrons.getParameter<edm::InputTag>("sumNeutralHadronEtHighThreshold");
    if( electrons.exists("sumPhotonEtHighThreshold") ) e_conf.sumPhotonEtHighThreshold = electrons.getParameter<edm::InputTag>("sumPhotonEtHighThreshold");
    if( electrons.exists("sumPUPt") ) e_conf.sumPUPt = electrons.getParameter<edm::InputTag>("sumPUPt");    
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if( photons.exists("photonSrc") ) ph_conf.photon_src = photons.getParameter<edm::InputTag>("photonSrc");    
    if( photons.exists("chargedHadronIso") ) ph_conf.chargedHadronIso = photons.getParameter<edm::InputTag>("chargedHadronIso");
    if( photons.exists("chargedHadronIsoWrongVtx") ) ph_conf.chargedHadronIsoWrongVtx = photons.getParameter<edm::InputTag>("chargedHadronIsoWrongVtx");
    if( photons.exists("neutralHadronIso") ) ph_conf.neutralHadronIso = photons.getParameter<edm::InputTag>("neutralHadronIso");
    if( photons.exists("photonIso") ) ph_conf.photonIso = photons.getParameter<edm::InputTag>("photonIso");
    if( photons.exists("modFrixione") ) ph_conf.modFrixione = photons.getParameter<edm::InputTag>("modFrixione");
    if( photons.exists("sumChargedParticlePt") ) ph_conf.sumChargedParticlePt = photons.getParameter<edm::InputTag>("sumChargedParticlePt");
    if( photons.exists("sumNeutralHadronEtHighThreshold") ) ph_conf.sumNeutralHadronEtHighThreshold = photons.getParameter<edm::InputTag>("sumNeutralHadronEtHighThreshold");
    if( photons.exists("sumPhotonEtHighThreshold") ) ph_conf.sumPhotonEtHighThreshold = photons.getParameter<edm::InputTag>("sumPhotonEtHighThreshold");
    if( photons.exists("sumPUPt") ) ph_conf.sumPUPt = photons.getParameter<edm::InputTag>("sumPUPt");
    
  }
  
  ele_idx = pho_idx = 0;
}

inline void get_product(const edm::Event& evt,
                        const edm::EDGetTokenT<edm::ValueMap<float> >& tok,
                        std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float> > >& map) {
  if( !tok.isUninitialized() ) evt.getByToken(tok,map[tok.index()]);
}

void EGPfIsolationModifierFromValueMaps::
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

  get_product(evt,e_conf.tok_sumChargedHadronPt,ele_vmaps);
  get_product(evt,e_conf.tok_sumNeutralHadronEt,ele_vmaps);
  get_product(evt,e_conf.tok_sumPhotonEt,ele_vmaps);
  get_product(evt,e_conf.tok_sumChargedParticlePt,ele_vmaps);
  get_product(evt,e_conf.tok_sumNeutralHadronEtHighThreshold,ele_vmaps);
  get_product(evt,e_conf.tok_sumPhotonEtHighThreshold,ele_vmaps);
  get_product(evt,e_conf.tok_sumPUPt,ele_vmaps);
  

  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);

    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[i] = ptr;
    }
  }
  
  get_product(evt,ph_conf.tok_chargedHadronIso,pho_vmaps);
  get_product(evt,ph_conf.tok_chargedHadronIsoWrongVtx,pho_vmaps);
  get_product(evt,ph_conf.tok_neutralHadronIso,pho_vmaps);
  get_product(evt,ph_conf.tok_photonIso,pho_vmaps);
  get_product(evt,ph_conf.tok_modFrixione,pho_vmaps);
  get_product(evt,ph_conf.tok_sumChargedParticlePt,pho_vmaps);
  get_product(evt,ph_conf.tok_sumNeutralHadronEtHighThreshold,pho_vmaps);
  get_product(evt,ph_conf.tok_sumPhotonEtHighThreshold,pho_vmaps);
  get_product(evt,ph_conf.tok_sumPUPt,pho_vmaps);
}

void EGPfIsolationModifierFromValueMaps::
setEventContent(const edm::EventSetup& evs) {
}

template<typename T, typename U, typename V>
inline void make_consumes(T& tag,U& tok,V& sume) { if( !(empty_tag == tag) ) tok = sume.template consumes<edm::ValueMap<float> >(tag); }

void EGPfIsolationModifierFromValueMaps::
setConsumes(edm::ConsumesCollector& sumes) {
  //setup electrons
  if( !(empty_tag == e_conf.electron_src) ) e_conf.tok_electron_src = sumes.consumes<edm::View<pat::Electron> >(e_conf.electron_src);  
  make_consumes(e_conf.sumChargedHadronPt,e_conf.tok_sumChargedHadronPt,sumes);
  make_consumes(e_conf.sumNeutralHadronEt,e_conf.tok_sumNeutralHadronEt,sumes);
  make_consumes(e_conf.sumPhotonEt,e_conf.tok_sumPhotonEt,sumes);
  make_consumes(e_conf.sumChargedParticlePt,e_conf.tok_sumChargedParticlePt,sumes);
  make_consumes(e_conf.sumNeutralHadronEtHighThreshold,e_conf.tok_sumNeutralHadronEtHighThreshold,sumes);
  make_consumes(e_conf.sumPhotonEtHighThreshold,e_conf.tok_sumPhotonEtHighThreshold,sumes);
  make_consumes(e_conf.sumPUPt,e_conf.tok_sumPUPt,sumes);

  // setup photons 
  if( !(empty_tag == ph_conf.photon_src) ) ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);
  make_consumes(ph_conf.chargedHadronIso,ph_conf.tok_chargedHadronIso,sumes);
  make_consumes(ph_conf.chargedHadronIsoWrongVtx,ph_conf.tok_chargedHadronIsoWrongVtx,sumes);
  make_consumes(ph_conf.neutralHadronIso,ph_conf.tok_neutralHadronIso,sumes);
  make_consumes(ph_conf.photonIso,ph_conf.tok_photonIso,sumes);
  make_consumes(ph_conf.modFrixione,ph_conf.tok_modFrixione,sumes);
  make_consumes(ph_conf.sumChargedParticlePt,ph_conf.tok_sumChargedParticlePt,sumes);
  make_consumes(ph_conf.sumNeutralHadronEtHighThreshold,ph_conf.tok_sumNeutralHadronEtHighThreshold,sumes);
  make_consumes(ph_conf.sumPhotonEtHighThreshold,ph_conf.tok_sumPhotonEtHighThreshold,sumes);
  make_consumes(ph_conf.sumPUPt,ph_conf.tok_sumPUPt,sumes);     
}

template<typename T, typename U, typename V>
inline void assignValue(const T& ptr, const U& tok, const V& map, float& value) {
  if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
}

void EGPfIsolationModifierFromValueMaps::
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
  auto pfIso = ele.pfIsolationVariables();
    
  assignValue(ptr,e_conf.tok_sumChargedHadronPt,ele_vmaps,pfIso.sumChargedHadronPt);
  assignValue(ptr,e_conf.tok_sumNeutralHadronEt,ele_vmaps,pfIso.sumNeutralHadronEt);
  assignValue(ptr,e_conf.tok_sumPhotonEt,ele_vmaps,pfIso.sumPhotonEt);
  assignValue(ptr,e_conf.tok_sumChargedParticlePt,ele_vmaps,pfIso.sumChargedParticlePt);
  assignValue(ptr,e_conf.tok_sumNeutralHadronEtHighThreshold,ele_vmaps,pfIso.sumNeutralHadronEtHighThreshold);
  assignValue(ptr,e_conf.tok_sumPhotonEtHighThreshold,ele_vmaps,pfIso.sumPhotonEtHighThreshold);
  assignValue(ptr,e_conf.tok_sumPUPt,ele_vmaps,pfIso.sumPUPt);
  
  ele.setPfIsolationVariables(pfIso);
  ++ele_idx;
}


void EGPfIsolationModifierFromValueMaps::
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
  auto pfIso = pho.getPflowIsolationVariables();
  assignValue(ptr,ph_conf.tok_chargedHadronIso,pho_vmaps,pfIso.chargedHadronIso);
  assignValue(ptr,ph_conf.tok_chargedHadronIsoWrongVtx,pho_vmaps,pfIso.chargedHadronIsoWrongVtx);
  assignValue(ptr,ph_conf.tok_neutralHadronIso,pho_vmaps,pfIso.neutralHadronIso);
  assignValue(ptr,ph_conf.tok_photonIso,pho_vmaps,pfIso.photonIso);
  assignValue(ptr,ph_conf.tok_modFrixione,pho_vmaps,pfIso.modFrixione);
  assignValue(ptr,ph_conf.tok_sumChargedParticlePt,pho_vmaps,pfIso.sumChargedParticlePt);
  assignValue(ptr,ph_conf.tok_sumNeutralHadronEtHighThreshold,pho_vmaps,pfIso.sumNeutralHadronEtHighThreshold);
  assignValue(ptr,ph_conf.tok_sumPhotonEtHighThreshold,pho_vmaps,pfIso.sumPhotonEtHighThreshold);
  assignValue(ptr,ph_conf.tok_sumPUPt,pho_vmaps,pfIso.sumPUPt);
  
  pho.setPflowIsolationVariables(pfIso);
  ++pho_idx;
}
