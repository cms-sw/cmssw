#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <tuple>
#include <array>

namespace {
  const edm::EDGetTokenT<edm::ValueMap<float> > empty_token;
  const edm::InputTag empty_tag("");

  const std::array<std::string,7>  electron_vars = { { "sumChargedHadronPt",
                                                              "sumNeutralHadronEt",
                                                              "sumPhotonEt",
                                                              "sumChargedParticlePt",
                                                              "sumNeutralHadronEtHighThreshold",
                                                              "sumPhotonEtHighThreshold",
                                                              "sumPUPt" } };
  
  const std::array<std::string,9> photon_vars = { { "chargedHadronIso",
                                                           "chargedHadronIsoWrongVtx",
                                                           "neutralHadronIso",
                                                           "photonIso",
                                                           "modFrixione",
                                                           "sumChargedParticlePt",
                                                           "sumNeutralHadronEtHighThreshold",
                                                           "sumPhotonEtHighThreshold",
                                                           "sumPUPt" } };
}

#include <unordered_map>

class EGPfIsolationModifierFromValueMaps : public ModifyObjectValueBase {
public:
  typedef std::unordered_map<std::string, edm::EDGetTokenT<edm::ValueMap<float> > > input_map;
  
  struct electron_config {
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    input_map electron_inputs;
  };
  
  struct photon_config {
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    input_map photon_inputs;
  };

  EGPfIsolationModifierFromValueMaps(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);
  
  void setEvent(const edm::Event&) final;
  
  void modifyObject(pat::Electron&) const final;
  void modifyObject(pat::Photon&) const final;

private:
  electron_config e_conf;
  photon_config   ph_conf;
  std::vector<edm::Ptr<reco::GsfElectron>> eles_by_oop; // indexed by original object ptr
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > ele_vmaps;
  std::vector<edm::Ptr<reco::Photon>> phos_by_oop;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > pho_vmaps;
  mutable unsigned ele_idx,pho_idx; // hack here until we figure out why some slimmedPhotons don't have original object ptrs
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGPfIsolationModifierFromValueMaps,
		  "EGPfIsolationModifierFromValueMaps");

EGPfIsolationModifierFromValueMaps::
EGPfIsolationModifierFromValueMaps(const edm::ParameterSet& conf, edm::ConsumesCollector& cc) :
  ModifyObjectValueBase(conf) {
  if( conf.exists("electron_config") ) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if( electrons.exists("electronSrc") ) e_conf.tok_electron_src = cc.consumes<edm::View<pat::Electron>>(electrons.getParameter<edm::InputTag>("electronSrc"));
    for( const std::string& varname : electron_vars ) {
      if( electrons.exists(varname) ) {
        e_conf.electron_inputs[varname] = cc.consumes<edm::ValueMap<float>>(electrons.getParameter<edm::InputTag>(varname));
      }
    }      
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if( photons.exists("photonSrc") ) ph_conf.tok_photon_src = cc.consumes<edm::View<pat::Photon>>(photons.getParameter<edm::InputTag>("photonSrc"));
    for( const std::string& varname : photon_vars ) {
      if( photons.exists(varname) ) {
        ph_conf.photon_inputs[varname] = cc.consumes<edm::ValueMap<float>>(photons.getParameter<edm::InputTag>(varname));
      }
    }     
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
    
    eles_by_oop.resize(eles->size());
    std::copy(eles->ptrs().begin(), eles->ptrs().end(), eles_by_oop.begin());
  }

  for( const std::string& varname : electron_vars ) {
    auto& inputs = e_conf.electron_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    get_product(evt,inputs[varname],ele_vmaps);
  }  

  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);

    phos_by_oop.resize(phos->size());
    std::copy(phos->ptrs().begin(), phos->ptrs().end(), phos_by_oop.begin());
  }
  
  for( const std::string& varname : photon_vars ) {
    auto& inputs = ph_conf.photon_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    get_product(evt,inputs[varname],pho_vmaps);
  }   
}

namespace {
  template<typename T, typename U, typename V>
  inline void assignValue(const T& ptr, const U& input_map, const std::string& name, const V& map, float& value) {
    auto itr = input_map.find(name);
    if( itr == input_map.end() ) return;
    const auto& tok = itr->second;
    if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
  }
}

void EGPfIsolationModifierFromValueMaps::
modifyObject(pat::Electron& ele) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference    
  edm::Ptr<reco::Candidate> ptr(ele.originalObjectRef());

  // The calls to this function should be matched to the order of the electrons
  // in eles_by_oop. In case it is called too many times, it will throw thanks
  // to the use of std::vector<T>::at().
  if( !e_conf.tok_electron_src.isUninitialized() ) ptr = eles_by_oop.at(ele_idx);

  //now we go through and modify the objects using the valuemaps we read in
  auto pfIso = ele.pfIsolationVariables();
    
  const auto& e_inputs = e_conf.electron_inputs;

  assignValue(ptr,e_inputs,electron_vars[0],ele_vmaps,pfIso.sumChargedHadronPt);
  assignValue(ptr,e_inputs,electron_vars[1],ele_vmaps,pfIso.sumNeutralHadronEt);
  assignValue(ptr,e_inputs,electron_vars[2],ele_vmaps,pfIso.sumPhotonEt);
  assignValue(ptr,e_inputs,electron_vars[3],ele_vmaps,pfIso.sumChargedParticlePt);
  assignValue(ptr,e_inputs,electron_vars[4],ele_vmaps,pfIso.sumNeutralHadronEtHighThreshold);
  assignValue(ptr,e_inputs,electron_vars[5],ele_vmaps,pfIso.sumPhotonEtHighThreshold);
  assignValue(ptr,e_inputs,electron_vars[6],ele_vmaps,pfIso.sumPUPt);
  
  ele.setPfIsolationVariables(pfIso);
  ++ele_idx;
}


void EGPfIsolationModifierFromValueMaps::
modifyObject(pat::Photon& pho) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());

  // The calls to this function should be matched to the order of the electrons
  // in eles_by_oop. In case it is called too many times, it will throw thanks
  // to the use of std::vector<T>::at().
  if( !ph_conf.tok_photon_src.isUninitialized() ) ptr = phos_by_oop.at(pho_idx);

  //now we go through and modify the objects using the valuemaps we read in
  auto pfIso = pho.getPflowIsolationVariables();

  const auto& ph_inputs = ph_conf.photon_inputs;

  assignValue(ptr,ph_inputs,photon_vars[0],pho_vmaps,pfIso.chargedHadronIso);
  assignValue(ptr,ph_inputs,photon_vars[1],pho_vmaps,pfIso.chargedHadronIsoWrongVtx);
  assignValue(ptr,ph_inputs,photon_vars[2],pho_vmaps,pfIso.neutralHadronIso);
  assignValue(ptr,ph_inputs,photon_vars[3],pho_vmaps,pfIso.photonIso);
  assignValue(ptr,ph_inputs,photon_vars[4],pho_vmaps,pfIso.modFrixione);
  assignValue(ptr,ph_inputs,photon_vars[5],pho_vmaps,pfIso.sumChargedParticlePt);
  assignValue(ptr,ph_inputs,photon_vars[6],pho_vmaps,pfIso.sumNeutralHadronEtHighThreshold);
  assignValue(ptr,ph_inputs,photon_vars[7],pho_vmaps,pfIso.sumPhotonEtHighThreshold);
  assignValue(ptr,ph_inputs,photon_vars[8],pho_vmaps,pfIso.sumPUPt);
  
  pho.setPflowIsolationVariables(pfIso);
  ++pho_idx;
}
