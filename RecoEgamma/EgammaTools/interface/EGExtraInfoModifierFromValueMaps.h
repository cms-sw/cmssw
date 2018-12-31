#ifndef RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h
#define RecoEgamma_EgammaTools_EGExtraInfoModifierFromValueMaps_h

#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

//class: EGExtraInfoModifierFromValueMaps
//  
//this is a generalisation of EGExtraInfoModiferFromFloatValueMaps
//orginal author of EGExtraInfoModiferFromFloatValueMaps : L. Gray (FNAL) 
//although it has been changed so much it is now almost unrecognisable from the original version
//converter to templated version: S. Harper (RAL)
//moderniser: S. Harper (RAL)
//
//This class allows an data of an arbitrary type in a ValueMap for pat::Electrons or pat::Photons
//to be put in the pat::Electron/Photon as userData, userInt or userFloat
//
//It assumes that the object can be added via pat::PATObject::userData, see pat::PATObject for the 
//constraints here
//
//The class has two template arguements:
//  MapType : c++ type of the object stored in the value map
//  OutputType : c++ type of how you want to store it in the pat::PATObject
//               this exists so you can specialise int and float (and future exceptions) to use
//               pat::PATObject::userInt and pat::PATObject::userFloat
//               The specialisations are done by EGXtraModFromVMObjFiller::addValueToObject
//               
// MapType and OutputType do not have to be same (but are by default). This is useful as it allows
// things like bools to and unsigned ints to be converted to ints to be stored as  a userInt
// rather than having to go to the bother of setting up userData hooks for them


namespace egmodifier{
  class EGID{};//dummy class to be used as a template arguement 
}

//our little helper classes
namespace egmodifier {
  template<typename MapType>
  class ValueMapData {
  public:
    ValueMapData(std::string name,const edm::InputTag& tag):name_(std::move(name)),tag_(tag){}
    
    void setHandle(const edm::Event& evt){if(!token_.isUninitialized()) evt.getByToken(token_,handle_);}
    void setToken(edm::ConsumesCollector& cc){if(!tag_.label().empty()) token_ = cc.consumes<edm::ValueMap<MapType> >(tag_);}

    const std::string& name()const{return name_;}
    void isValid()const {return handle_.isValid();}    
    template<typename RefType>
    typename edm::ValueMap<MapType>::const_reference_type
    value(const RefType& ref)const{return (*handle_)[ref];}

  private:
    std::string name_;
    edm::InputTag tag_;
    edm::EDGetTokenT<edm::ValueMap<MapType> > token_;
    edm::Handle<edm::ValueMap<MapType> > handle_;
  };

  template<typename OutputType>
  class EGXtraModFromVMObjFiller {
  public:
    EGXtraModFromVMObjFiller()=delete;
    ~EGXtraModFromVMObjFiller()=delete;
    
    template<typename ObjType,typename MapType>
    static void 
    addValueToObject(ObjType& obj,
		     const ValueMapData<MapType>& vmData,
		     bool overrideExistingValues);
    
    template<typename ObjType,typename MapType>
    static void 
    addValuesToObject(ObjType& obj,
		      const std::vector<ValueMapData<MapType> >& vmapsData,
		      bool overrideExistingValues){
      for(auto& vmapData : vmapsData){
	addValueToObject(obj,vmapData,overrideExistingValues);
      }  
    }
    
    //will do a UserData add but specialisations exist for float and ints
    //in theory could do most of this with function pointers for the different addX and hasX functions
    //but specialisations are simplier
    template<typename ObjType>
    static void addValue(ObjType& obj,const std::string& name,const OutputType& value){obj.addUserData(name,value,true);}
    template<typename ObjType>
    static bool hasValue(ObjType& obj,const std::string& name){return obj.hasUserData(name);}
    
  };		    
}

template<typename MapType,typename OutputType=MapType>
class EGExtraInfoModifierFromValueMaps : public ModifyObjectValueBase {
public:
  EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf);
  
  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final{}
  void setConsumes(edm::ConsumesCollector&) final;
  
  void modifyObject(pat::Electron&) const final;
  void modifyObject(pat::Photon&) const final;
 
private:
  std::vector<egmodifier::ValueMapData<MapType> > eleVMData_;
  std::vector<egmodifier::ValueMapData<MapType> > phoVMData_;
  bool overrideExistingValues_;
};


template<typename MapType,typename OutputType>
EGExtraInfoModifierFromValueMaps<MapType,OutputType>::
EGExtraInfoModifierFromValueMaps(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {
  overrideExistingValues_ = conf.exists("overrideExistingValues") ? conf.getParameter<bool>("overrideExistingValues") : false;
  if( conf.exists("electron_config") ) {
    const edm::ParameterSet& ele_cfg = conf.getParameter<edm::ParameterSet>("electron_config");
    const std::vector<std::string>& parameters = ele_cfg.getParameterNames();
    for( const std::string& name : parameters ) {
      if( ele_cfg.existsAs<edm::InputTag>(name) ) {
        eleVMData_.emplace_back(egmodifier::ValueMapData<MapType>(name,ele_cfg.getParameter<edm::InputTag>(name)));
      }
    }    
  }
  if( conf.exists("photon_config") ) {
    const edm::ParameterSet& pho_cfg = conf.getParameter<edm::ParameterSet>("photon_config");
    const std::vector<std::string>& parameters = pho_cfg.getParameterNames();
    for( const std::string& name : parameters ) {
      if( pho_cfg.existsAs<edm::InputTag>(name) ) {
        phoVMData_.emplace_back(egmodifier::ValueMapData<MapType>(name,pho_cfg.getParameter<edm::InputTag>(name)));
      }
    }    
  }
}

template<typename MapType,typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType,OutputType>::
setEvent(const edm::Event& evt) {

  for( auto& data : eleVMData_) data.setHandle(evt);
  for( auto& data : phoVMData_) data.setHandle(evt);
  
}

template<typename MapType,typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType,OutputType>::
setConsumes(edm::ConsumesCollector& cc) {  
  for( auto& data : eleVMData_) data.setToken(cc);
  for( auto& data : phoVMData_) data.setToken(cc);
}

template<typename MapType,typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType,OutputType>::
modifyObject(pat::Electron& ele) const {
  egmodifier::EGXtraModFromVMObjFiller<OutputType>::addValuesToObject(ele,eleVMData_,overrideExistingValues_);
}

template<typename MapType,typename OutputType>
void EGExtraInfoModifierFromValueMaps<MapType,OutputType>::
modifyObject(pat::Photon& pho) const {
  egmodifier::EGXtraModFromVMObjFiller<OutputType>::addValuesToObject(pho,phoVMData_,overrideExistingValues_);
}

template<typename OutputType>
template<typename ObjType,typename MapType>
void egmodifier::EGXtraModFromVMObjFiller<OutputType>::
addValueToObject(ObjType& obj,
		 const egmodifier::ValueMapData<MapType>& mapData,
		 bool overrideExistingValues)
{
  if(obj.parentRefs().empty()){
    throw cms::Exception("LogicError") << " object "<<typeid(obj).name()<<" has no parent references, these should be set with ::addParentRef before calling the modifier";
  }
  auto ptr = obj.parentRefs().back();
  if( overrideExistingValues || !hasValue(obj,mapData.name()) ) {
    addValue(obj,mapData.name(),mapData.value(ptr));
  } else {
    throw cms::Exception("ValueNameAlreadyExists")
      << "Trying to add new UserData = " << mapData.name()
      << " failed because it already exists and you didnt specify to override it (set in the config overrideExistingValues=cms.bool(True) )";
  }
}


template<>
template<typename ObjType>
void egmodifier::EGXtraModFromVMObjFiller<float>::
addValue(ObjType& obj,const std::string& name,const float& value){obj.addUserFloat(name,value,true);}

template<>
template<typename ObjType>
bool egmodifier::EGXtraModFromVMObjFiller<float>::
hasValue(ObjType& obj,const std::string& name){return obj.hasUserFloat(name);}

template<>
template<typename ObjType>
void egmodifier::EGXtraModFromVMObjFiller<int>::
addValue(ObjType& obj,const std::string& name,const int& value){obj.addUserInt(name,value,true);}

template<>
template<typename ObjType>
bool egmodifier::EGXtraModFromVMObjFiller<int>::
hasValue(ObjType& obj,const std::string& name){return obj.hasUserInt(name);}

template<>
template<>
void egmodifier::EGXtraModFromVMObjFiller<egmodifier::EGID>::
addValuesToObject(pat::Electron& obj,
		  const std::vector<egmodifier::ValueMapData<float> >& vmapsData,
		  bool overrideExistingValues)
{
  std::vector<std::pair<std::string,float >> ids;
  if(obj.parentRefs().empty()){
    throw cms::Exception("LogicError") << " object "<<typeid(obj).name()<<" has no parent references, these should be set with ::addParentRef before calling the modifier";
  }    
  auto ptr = obj.parentRefs().back();
  for( auto& vmapData : vmapsData ) {
    float idVal = vmapData.value(ptr);
    ids.push_back({vmapData.name(),idVal});
  }   
  std::sort(ids.begin(),ids.end(),[](auto& lhs,auto& rhs){return lhs.first<rhs.first;});
  obj.setElectronIDs(ids);
}

template<>
template<>
void egmodifier::EGXtraModFromVMObjFiller<egmodifier::EGID>::
addValuesToObject(pat::Photon& obj,
		  const std::vector<egmodifier::ValueMapData<float> >& vmapsData,
		  bool overrideExistingValues)
{
  //we do a float->bool conversion here to make things easier to be consistent with electrons
  std::vector<std::pair<std::string,bool> > ids;
  if(obj.parentRefs().empty()){
    throw cms::Exception("LogicError") << " object "<<typeid(obj).name()<<" has no parent references, these should be set with ::addParentRef before calling the modifier";
  }
  auto ptr = obj.parentRefs().back();
  for( auto& vmapData : vmapsData ) {
    float idVal = vmapData.value(ptr);
    ids.push_back({vmapData.name(),idVal});
  }  
  std::sort(ids.begin(),ids.end(),[](auto& lhs,auto& rhs){return lhs.first<rhs.first;});
  obj.setPhotonIDs(ids);
}

#endif
