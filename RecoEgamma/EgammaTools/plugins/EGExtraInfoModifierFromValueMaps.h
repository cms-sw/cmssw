#ifndef RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h
#define RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h

#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace {
  const edm::EDGetTokenT<edm::ValueMap<float> > empty_token;
  const edm::InputTag empty_tag;
}

#include <unordered_map>

//this is a generalisation of EGExtraInfoModiferFromFloatValueMaps
//orginal author of EGExtraInfoModiferFromFloatValueMaps : L. Gray (FNAL)
//converter to templated version: S. Harper (RAL)
template<typename DataType>
class EGExtraInfoModifierFromValueMaps : public ModifyObjectValueBase {
public:
  typedef edm::EDGetTokenT<edm::ValueMap<DataType> > ValMapToken;
  typedef std::unordered_map<std::string,ValMapToken> ValueMaps;
  typedef std::unordered_map<std::string,edm::InputTag> ValueMapsTags;
  struct electron_config {
    edm::InputTag electron_src;
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    ValueMapsTags valuemaps;
    ValueMaps tok_valuemaps;    
  };

  struct photon_config {
    edm::InputTag photon_src;
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    ValueMapsTags valuemaps;
    ValueMaps tok_valuemaps;  
  };

  EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf);
  
  void setEvent(const edm::Event&) override final;
  void setEventContent(const edm::EventSetup&) override final;
  void setConsumes(edm::ConsumesCollector&) override final;
  
  void modifyObject(pat::Electron&) const override final;
  void modifyObject(pat::Photon&) const override final;

private:  
  //will do a UserData add but specialisations exist for float and ints
  template<typename ObjType>
  void addValueToObject(ObjType& pho,
			const edm::Ptr<reco::Candidate>& ptr,
			const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<DataType> > >& vmaps,
			const std::pair<std::string,ValMapToken>& val_map)const;

private:
  electron_config e_conf;
  photon_config   ph_conf;
  std::unordered_map<unsigned,edm::Ptr<reco::GsfElectron> > eles_by_oop; // indexed by original object ptr
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<DataType> > > ele_vmaps;
  std::unordered_map<unsigned,edm::Ptr<reco::Photon> > phos_by_oop;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<DataType> > > pho_vmaps;
  mutable unsigned ele_idx,pho_idx; // hack here until we figure out why some slimmedPhotons don't have original object ptrs
};


template<typename DataType>
EGExtraInfoModifierFromValueMaps<DataType>::
EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {
  constexpr char electronSrc[] =  "electronSrc";
  constexpr char photonSrc[] =  "photonSrc";

  if( conf.exists("electron_config") ) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if( electrons.exists(electronSrc) ) e_conf.electron_src = electrons.getParameter<edm::InputTag>(electronSrc);
    const std::vector<std::string> parameters = electrons.getParameterNames();
    for( const std::string& name : parameters ) {
      if( std::string(electronSrc) == name ) continue;
      if( electrons.existsAs<edm::InputTag>(name) ) {
        e_conf.valuemaps[name] = electrons.getParameter<edm::InputTag>(name);
      }
    }    
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if( photons.exists(photonSrc) ) ph_conf.photon_src = photons.getParameter<edm::InputTag>(photonSrc);
    const std::vector<std::string> parameters = photons.getParameterNames();
    for( const std::string& name : parameters ) {
      if( std::string(photonSrc) == name ) continue;
      if( photons.existsAs<edm::InputTag>(name) ) {
        ph_conf.valuemaps[name] = photons.getParameter<edm::InputTag>(name);
      }
    } 
  }
  ele_idx = pho_idx = 0;
}

namespace {
  template<typename T>
  inline void get_product(const edm::Event& evt,
                          const edm::EDGetTokenT<edm::ValueMap<T> >& tok,
                          std::unordered_map<unsigned, edm::Handle<edm::ValueMap<T> > >& map) {
    evt.getByToken(tok,map[tok.index()]);
  }
}

template<typename DataType>
void EGExtraInfoModifierFromValueMaps<DataType>::
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

  for( auto itr = e_conf.tok_valuemaps.begin(); itr != e_conf.tok_valuemaps.end(); ++itr ) {
    get_product(evt,itr->second,ele_vmaps);
  }

  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);

    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[i] = ptr;
    }
  }

  for( auto itr = ph_conf.tok_valuemaps.begin(); itr != ph_conf.tok_valuemaps.end(); ++itr ) {
    get_product(evt,itr->second,pho_vmaps);
  }
}


template<typename DataType>
void EGExtraInfoModifierFromValueMaps<DataType>::
setEventContent(const edm::EventSetup& evs) {
}

namespace {
  template<typename T>
  inline void make_consumes(const edm::InputTag& tag,edm::EDGetTokenT<T>& token,
			    edm::ConsumesCollector& cc)
  { if( !(empty_tag == tag) ) token = cc.consumes<T>(tag); }
}

template<typename DataType>
void EGExtraInfoModifierFromValueMaps<DataType>::
setConsumes(edm::ConsumesCollector& sumes) {
  //setup electrons
  if( !(empty_tag == e_conf.electron_src) ) e_conf.tok_electron_src = sumes.consumes<edm::View<pat::Electron> >(e_conf.electron_src);  

  for( auto itr = e_conf.valuemaps.begin(); itr != e_conf.valuemaps.end(); ++itr ) {
    make_consumes(itr->second,e_conf.tok_valuemaps[itr->first],sumes);
  }
  
  // setup photons 
  if( !(empty_tag == ph_conf.photon_src) ) ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);
  
  for( auto itr = ph_conf.valuemaps.begin(); itr != ph_conf.valuemaps.end(); ++itr ) {
    make_consumes(itr->second,ph_conf.tok_valuemaps[itr->first],sumes);
  }
}

namespace {
  template<typename T, typename U, typename V, typename DataType >
  inline void assignValue(const T& ptr, const U& tok, const V& map, DataType& value) {
    if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
  }
}

template<typename DataType>
void EGExtraInfoModifierFromValueMaps<DataType>::
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
        << "Original object pointer with key = " << ele.originalObjectRef().key() 
        << " not found in cache!";
    }
  }
  //now we go through and modify the objects using the valuemaps we read in
  for( auto itr = e_conf.tok_valuemaps.begin(); itr != e_conf.tok_valuemaps.end(); ++itr ) {
    addValueToObject(ele,ptr,ele_vmaps,*itr);
  }  
  ++ele_idx;
}

template<typename DataType>
void EGExtraInfoModifierFromValueMaps<DataType>::
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
  for( auto itr = ph_conf.tok_valuemaps.begin(); itr != ph_conf.tok_valuemaps.end(); ++itr ) {
    addValueToObject(pho,ptr,pho_vmaps,*itr);
  }    
  ++pho_idx;
}


template<typename DataType>
template<typename ObjType>
void EGExtraInfoModifierFromValueMaps<DataType>::
addValueToObject(ObjType& obj,
		 const edm::Ptr<reco::Candidate>& ptr,
		 const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<DataType> > >& vmaps,
		 const std::pair<std::string,EGExtraInfoModifierFromValueMaps::ValMapToken> & val_map)const
{
  DataType value;
  assignValue(ptr,val_map.second,vmaps,value);
  if( !obj.hasUserData(val_map.first) ) {
    obj.addUserData(val_map.first,value);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserData = " << val_map.first
      << " failed because it already exists!";
  }
}  
template<>
template<typename ObjType>
void EGExtraInfoModifierFromValueMaps<float>::
addValueToObject(ObjType& obj,
		 const edm::Ptr<reco::Candidate>& ptr,
		 const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > >& vmaps,
		 const std::pair<std::string,EGExtraInfoModifierFromValueMaps::ValMapToken> & val_map)const
{
  float value(0.0);
  assignValue(ptr,val_map.second,vmaps,value);
  if( !obj.hasUserFloat(val_map.first) ) {
    obj.addUserFloat(val_map.first,value);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserFloat = " << val_map.first
      << " failed because it already exists!";
  }
}

template<>
template<typename ObjType>
void EGExtraInfoModifierFromValueMaps<int>::
addValueToObject(ObjType& obj,
		 const edm::Ptr<reco::Candidate>& ptr,
		 const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<int> > >& vmaps,
		 const std::pair<std::string,EGExtraInfoModifierFromValueMaps::ValMapToken> & val_map)const
{
  int value(0);
  assignValue(ptr,val_map.second,vmaps,value);
  if( !obj.hasUserInt(val_map.first) ) {
    obj.addUserInt(val_map.first,value);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserInt = " << val_map.first
      << " failed because it already exists!";
  }
}  

template<>
template<typename ObjType>
void EGExtraInfoModifierFromValueMaps<bool>::
addValueToObject(ObjType& obj,
		 const edm::Ptr<reco::Candidate>& ptr,
		 const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<bool> > >& vmaps,
		 const std::pair<std::string,EGExtraInfoModifierFromValueMaps::ValMapToken> & val_map)const
{
  bool value(0);
  assignValue(ptr,val_map.second,vmaps,value);
  if( !obj.hasUserInt(val_map.first) ) {
    obj.addUserInt(val_map.first,value);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserInt (from a bool) = " << val_map.first
      << " failed because it already exists!";
  }
} 
template<>
template<typename ObjType>
void EGExtraInfoModifierFromValueMaps<unsigned int>::
addValueToObject(ObjType& obj,
		 const edm::Ptr<reco::Candidate>& ptr,
		 const std::unordered_map<unsigned,edm::Handle<edm::ValueMap<unsigned int> > >& vmaps,
		 const std::pair<std::string,EGExtraInfoModifierFromValueMaps::ValMapToken> & val_map)const
{
  unsigned int value(0);
  assignValue(ptr,val_map.second,vmaps,value);
  if( !obj.hasUserInt(val_map.first) ) {
    obj.addUserInt(val_map.first,value);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserInt (from an unsigned int) = " << val_map.first
      << " failed because it already exists!";
  }
} 
 

#endif
