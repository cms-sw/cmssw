#ifndef RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h
#define RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h

#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include <unordered_map>

namespace {
  const edm::InputTag empty_tag;
}

//class: EGExtraInfoModifierFromValueMaps
//
//this is a generalisation of EGExtraInfoModiferFromFloatValueMaps
//orginal author of EGExtraInfoModiferFromFloatValueMaps : L. Gray (FNAL)
//converter to templated version: S. Harper (RAL)
//
//This class allows an data of an arbitrary type in a ValueMap for pat::Electrons or pat::Photons
//to be put in the pat::Electron/Photon as userData, userInt or userFloat
//
//IMPORTANT INFO:
//by default the ValueMap is keyed to the object the pat::Electron/Photon was created from
//if you want to use a ValueMap which is keyed to a different collection (ie perhaps the same
//as the electrons you are aring, you must set "electronSrc" and "photonSrc" inside the ele/pho configs
//so if you are running over slimmedElectrons and want to read a ValueMap keyed to slimmedElectrons
//you need to set "electron_config.electronSrc = cms.InputTag("slimmedElectrons")"
//
//It assumes that the object can be added via pat::PATObject::userData, see pat::PATObject for the
//constraints here
//
//The class has two template arguements:
//  MapType : c++ type of the object stored in the value mape
//  OutputType : c++ type of how you want to store it in the pat::PATObject
//               this exists so you can specialise int and float (and future exceptions) to use
//               pat::PATObject::userInt and pat::PATObject::userFloat
//               The specialisations are done by EGXtraModFromVMObjFiller::addValueToObject
//
// MapType and OutputType do not have to be same (but are by default). This is useful as it allows
// things like bools to and unsigned ints to be converted to ints to be stored as  a userInt
// rather than having to go to the bother of setting up userData hooks for them

namespace egmodifier {
  class EGID {};  //dummy class to be used as a template arguement
}  // namespace egmodifier

template <typename OutputType>
class EGXtraModFromVMObjFiller {
public:
  EGXtraModFromVMObjFiller() = delete;
  ~EGXtraModFromVMObjFiller() = delete;

  //will do a UserData add but specialisations exist for float and ints
  template <typename ObjType, typename MapType>
  static void addValueToObject(ObjType& obj,
                               const edm::Ptr<reco::Candidate>& ptr,
                               const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>>& vmaps,
                               const std::pair<const std::string, edm::EDGetTokenT<edm::ValueMap<MapType>>>& val_map,
                               bool overrideExistingValues);

  template <typename ObjType, typename MapType>
  static void addValuesToObject(
      ObjType& obj,
      const edm::Ptr<reco::Candidate>& ptr,
      const std::unordered_map<std::string, edm::EDGetTokenT<edm::ValueMap<MapType>>>& vmaps_token,
      const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>>& vmaps,
      bool overrideExistingValues) {
    for (auto itr = vmaps_token.begin(); itr != vmaps_token.end(); ++itr) {
      addValueToObject(obj, ptr, vmaps, *itr, overrideExistingValues);
    }
  }
};

template <typename MapType, typename OutputType = MapType>
class EGExtraInfoModifierFromValueMaps : public ModifyObjectValueBase {
public:
  using ValMapToken = edm::EDGetTokenT<edm::ValueMap<MapType>>;
  using ValueMaps = std::unordered_map<std::string, ValMapToken>;
  struct electron_config {
    edm::EDGetTokenT<edm::View<reco::GsfElectron>> tok_electron_src;
    ValueMaps tok_valuemaps;
  };

  struct photon_config {
    edm::EDGetTokenT<edm::View<reco::Photon>> tok_photon_src;
    ValueMaps tok_valuemaps;
  };

  EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);

  void setEvent(const edm::Event&) final;

  void modifyObject(pat::Electron&) const final;
  void modifyObject(pat::Photon&) const final;

private:
  electron_config e_conf;
  photon_config ph_conf;
  std::vector<edm::Ptr<reco::GsfElectron>> eles_by_oop;  // indexed by original object ptr
  std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>> ele_vmaps;
  std::vector<edm::Ptr<reco::Photon>> phos_by_oop;
  std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>> pho_vmaps;
  mutable unsigned ele_idx,
      pho_idx;  // hack here until we figure out why some slimmedPhotons don't have original object ptrs
  bool overrideExistingValues_;
};

template <typename MapType, typename OutputType>
EGExtraInfoModifierFromValueMaps<MapType, OutputType>::EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf,
                                                                                        edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf) {
  constexpr char electronSrc[] = "electronSrc";
  constexpr char photonSrc[] = "photonSrc";
  overrideExistingValues_ =
      conf.exists("overrideExistingValues") ? conf.getParameter<bool>("overrideExistingValues") : false;
  if (conf.exists("electron_config")) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if (electrons.exists(electronSrc))
      e_conf.tok_electron_src =
          cc.consumes<edm::View<reco::GsfElectron>>(electrons.getParameter<edm::InputTag>(electronSrc));

    const std::vector<std::string> parameters = electrons.getParameterNames();
    for (const std::string& name : parameters) {
      if (std::string(electronSrc) == name)
        continue;
      if (electrons.existsAs<edm::InputTag>(name)) {
        e_conf.tok_valuemaps[name] = cc.consumes<edm::ValueMap<MapType>>(electrons.getParameter<edm::InputTag>(name));
      }
    }
  }
  if (conf.exists("photon_config")) {
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if (photons.exists(photonSrc))
      ph_conf.tok_photon_src = cc.consumes<edm::View<reco::Photon>>(photons.getParameter<edm::InputTag>(photonSrc));
    const std::vector<std::string> parameters = photons.getParameterNames();
    for (const std::string& name : parameters) {
      if (std::string(photonSrc) == name)
        continue;
      if (photons.existsAs<edm::InputTag>(name)) {
        ph_conf.tok_valuemaps[name] = cc.consumes<edm::ValueMap<MapType>>(photons.getParameter<edm::InputTag>(name));
      }
    }
  }
  ele_idx = pho_idx = 0;
}

template <typename MapType, typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType, OutputType>::setEvent(const edm::Event& evt) {
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

  for (auto const& itr : e_conf.tok_valuemaps) {
    ele_vmaps[itr.second.index()] = evt.getHandle(itr.second);
  }

  if (!ph_conf.tok_photon_src.isUninitialized()) {
    auto phos = evt.getHandle(ph_conf.tok_photon_src);

    phos_by_oop.resize(phos->size());
    std::copy(phos->ptrs().begin(), phos->ptrs().end(), phos_by_oop.begin());
  }

  for (auto const& itr : ph_conf.tok_valuemaps) {
    pho_vmaps[itr.second.index()] = evt.getHandle(itr.second);
  }
}

namespace {
  template <typename T, typename U, typename V, typename MapType>
  inline void assignValue(const T& ptr, const U& tok, const V& map, MapType& value) {
    if (!tok.isUninitialized())
      value = map.find(tok.index())->second->get(ptr.id(), ptr.key());
  }
}  // namespace

template <typename MapType, typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType, OutputType>::modifyObject(pat::Electron& ele) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(ele.originalObjectRef());
  if (!e_conf.tok_electron_src.isUninitialized())
    ptr = eles_by_oop.at(ele_idx);
  //now we go through and modify the objects using the valuemaps we read in
  EGXtraModFromVMObjFiller<OutputType>::addValuesToObject(
      ele, ptr, e_conf.tok_valuemaps, ele_vmaps, overrideExistingValues_);
  ++ele_idx;
}

template <typename MapType, typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType, OutputType>::modifyObject(pat::Photon& pho) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());
  if (!ph_conf.tok_photon_src.isUninitialized())
    ptr = phos_by_oop.at(pho_idx);
  //now we go through and modify the objects using the valuemaps we read in
  EGXtraModFromVMObjFiller<OutputType>::addValuesToObject(
      pho, ptr, ph_conf.tok_valuemaps, pho_vmaps, overrideExistingValues_);

  ++pho_idx;
}

template <typename OutputType>
template <typename ObjType, typename MapType>
void EGXtraModFromVMObjFiller<OutputType>::addValueToObject(
    ObjType& obj,
    const edm::Ptr<reco::Candidate>& ptr,
    const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>>& vmaps,
    const std::pair<const std::string, edm::EDGetTokenT<edm::ValueMap<MapType>>>& val_map,
    bool overrideExistingValues) {
  MapType value{};
  assignValue(ptr, val_map.second, vmaps, value);
  if (overrideExistingValues || !obj.hasUserData(val_map.first)) {
    obj.addUserData(val_map.first, value, true);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
        << "Trying to add new UserData = " << val_map.first
        << " failed because it already exists and you didnt specify to override it (set in the config "
           "overrideExistingValues=cms.bool(True) )";
  }
}

template <>
template <typename ObjType, typename MapType>
void EGXtraModFromVMObjFiller<float>::addValueToObject(
    ObjType& obj,
    const edm::Ptr<reco::Candidate>& ptr,
    const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>>& vmaps,
    const std::pair<const std::string, edm::EDGetTokenT<edm::ValueMap<MapType>>>& val_map,
    bool overrideExistingValues) {
  float value(0.0);
  assignValue(ptr, val_map.second, vmaps, value);
  if (overrideExistingValues || !obj.hasUserFloat(val_map.first)) {
    obj.addUserFloat(val_map.first, value, true);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
        << "Trying to add new UserFloat = " << val_map.first
        << " failed because it already exists and you didnt specify to override it (set in the config "
           "overrideExistingValues=cms.bool(True) )";
  }
}

template <>
template <typename ObjType, typename MapType>
void EGXtraModFromVMObjFiller<int>::addValueToObject(
    ObjType& obj,
    const edm::Ptr<reco::Candidate>& ptr,
    const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<MapType>>>& vmaps,
    const std::pair<const std::string, edm::EDGetTokenT<edm::ValueMap<MapType>>>& val_map,
    bool overrideExistingValues) {
  int value(0);
  assignValue(ptr, val_map.second, vmaps, value);
  if (overrideExistingValues || !obj.hasUserInt(val_map.first)) {
    obj.addUserInt(val_map.first, value, true);
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
        << "Trying to add new UserInt = " << val_map.first
        << " failed because it already exists and you didnt specify to override it (set in the config "
           "overrideExistingValues=cms.bool(True) )";
  }
}

template <>
template <>
inline void EGXtraModFromVMObjFiller<egmodifier::EGID>::addValuesToObject(
    pat::Electron& obj,
    const edm::Ptr<reco::Candidate>& ptr,
    const std::unordered_map<std::string, edm::EDGetTokenT<edm::ValueMap<float>>>& vmaps_token,
    const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float>>>& vmaps,
    bool overrideExistingValues) {
  std::vector<std::pair<std::string, float>> ids;
  for (auto itr = vmaps_token.begin(); itr != vmaps_token.end(); ++itr) {
    float idVal(0);
    assignValue(ptr, itr->second, vmaps, idVal);
    ids.push_back({itr->first, idVal});
  }
  std::sort(ids.begin(), ids.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });
  obj.setElectronIDs(ids);
}

template <>
template <>
inline void EGXtraModFromVMObjFiller<egmodifier::EGID>::addValuesToObject(
    pat::Photon& obj,
    const edm::Ptr<reco::Candidate>& ptr,
    const std::unordered_map<std::string, edm::EDGetTokenT<edm::ValueMap<float>>>& vmaps_token,
    const std::unordered_map<unsigned, edm::Handle<edm::ValueMap<float>>>& vmaps,
    bool overrideExistingValues) {
  //we do a float->bool conversion here to make things easier to be consistent with electrons
  std::vector<std::pair<std::string, bool>> ids;
  for (auto itr = vmaps_token.begin(); itr != vmaps_token.end(); ++itr) {
    float idVal(0);
    assignValue(ptr, itr->second, vmaps, idVal);
    ids.push_back({itr->first, idVal});
  }
  std::sort(ids.begin(), ids.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });
  obj.setPhotonIDs(ids);
}

#endif
