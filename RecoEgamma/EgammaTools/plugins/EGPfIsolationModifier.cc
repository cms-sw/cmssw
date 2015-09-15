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
  const static edm::InputTag empty_tag("");

  const static std::array<std::string,7>  electron_vars = { { "sumChargedHadronPt",
                                                              "sumNeutralHadronEt",
                                                              "sumPhotonEt",
                                                              "sumChargedParticlePt",
                                                              "sumNeutralHadronEtHighThreshold",
                                                              "sumPhotonEtHighThreshold",
                                                              "sumPUPt" } };
  
  const static std::array<std::string,9> photon_vars = { { "chargedHadronIso",
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
  typedef std::tuple<edm::InputTag,edm::EDGetTokenT<edm::ValueMap<float> > > tag_and_token;
  typedef std::unordered_map<std::string,tag_and_token> input_map;
  
  struct electron_config {
    edm::InputTag electron_src;    
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    input_map electron_inputs;
  };
  
  struct photon_config {
    edm::InputTag photon_src;    
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    input_map photon_inputs;
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
    for( const std::string& varname : electron_vars ) {
      if( electrons.exists(varname) ) {
        std::get<0>(e_conf.electron_inputs[varname]) = electrons.getParameter<edm::InputTag>(varname);
      }
    }      
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if( photons.exists("photonSrc") ) ph_conf.photon_src = photons.getParameter<edm::InputTag>("photonSrc");  
    for( const std::string& varname : photon_vars ) {
      if( photons.exists(varname) ) {
        std::get<0>(ph_conf.photon_inputs[varname]) = photons.getParameter<edm::InputTag>(varname);
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
    
    for( unsigned i = 0; i < eles->size(); ++i ) {
      edm::Ptr<pat::Electron> ptr = eles->ptrAt(i);
      eles_by_oop[i] = ptr;
    }
  }

  for( const std::string& varname : electron_vars ) {
    auto& inputs = e_conf.electron_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    get_product(evt,std::get<1>(inputs[varname]),ele_vmaps);
  }  

  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);

    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[i] = ptr;
    }
  }
  
  for( const std::string& varname : photon_vars ) {
    auto& inputs = ph_conf.photon_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    get_product(evt,std::get<1>(inputs[varname]),pho_vmaps);
  }   
}

void EGPfIsolationModifierFromValueMaps::
setEventContent(const edm::EventSetup& evs) {
}

namespace {
  template<typename T, typename U, typename V>
  inline void make_consumes(T& tag,U& tok,V& sume) { if( !(empty_tag == tag) ) tok = sume.template consumes<edm::ValueMap<float> >(tag); }
}

void EGPfIsolationModifierFromValueMaps::
setConsumes(edm::ConsumesCollector& sumes) {
  //setup electrons
  if( !(empty_tag == e_conf.electron_src) ) e_conf.tok_electron_src = sumes.consumes<edm::View<pat::Electron> >(e_conf.electron_src);  
  
  for( const std::string& varname : electron_vars ) {
    auto& inputs = e_conf.electron_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    auto& the_tuple = inputs[varname];
    make_consumes(std::get<0>(the_tuple),std::get<1>(the_tuple),sumes);
  }  

  // setup photons 
  if( !(empty_tag == ph_conf.photon_src) ) ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);

  for( const std::string& varname : photon_vars ) {
    auto& inputs = ph_conf.photon_inputs;
    if( inputs.find(varname) == inputs.end() ) continue;
    auto& the_tuple = inputs[varname];
    make_consumes(std::get<0>(the_tuple),std::get<1>(the_tuple),sumes);
  }   
}

namespace {
  template<typename T, typename U, typename V>
  inline void assignValue(const T& ptr, const U& input_map, const std::string& name, const V& map, float& value) {
    auto itr = input_map.find(name);
    if( itr == input_map.end() ) return;
    const auto& tok = std::get<1>(itr->second);
    if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
  }
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
