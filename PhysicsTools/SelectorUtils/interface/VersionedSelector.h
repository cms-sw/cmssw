#ifndef PhysicsTools_SelectorUtils_VersionedSelector_h
#define PhysicsTools_SelectorUtils_VersionedSelector_h

/**
  \class    VersionedSelector VersionedSelector.h "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
  \brief    cut-flow versioning info in the event provenance

  class template to implement versioning for IDs that's available in the 
  event provenance or available by hash-code in the event record

  \author Lindsey Gray
*/



#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/SelectorUtils/interface/CandidateCut.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

// because we need to be able to validate the ID
#include <openssl/md5.h>

#include <memory>

namespace candf = candidate_functions;

template<class T>
class VersionedSelector : public Selector<T> {
 public:
 VersionedSelector() : Selector<T>(), initialized_(false) {}

 VersionedSelector(const edm::ParameterSet& conf) : 
  Selector<T>(),
  initialized_(false) { 
    constexpr unsigned length = MD5_DIGEST_LENGTH;
    edm::ParameterSet trackedPart = conf.trackedPart();
    name_ = conf.getParameter<std::string>("idName");
    memset(id_md5_,0,length*sizeof(unsigned char));
    std::string tracked(trackedPart.dump()), untracked(conf.dump());   
    if ( tracked != untracked ) {
      throw cms::Exception("InvalidConfiguration")
	<< "VersionedSelector does not allow untracked parameters"
	<< " in the configuration ParameterSet!";
    }
    // now setup the md5 and cute accessor functions
    MD5((unsigned char*)tracked.c_str(), tracked.size(), id_md5_);
    char buf[32];
    for( unsigned i=0; i<MD5_DIGEST_LENGTH; ++i ){
      sprintf(buf, "%02x", id_md5_[i]);
      md5_string_.append( buf );
    }
    initialize(conf);
    this->retInternal_  = this->getBitTemplate();
  }
  
  virtual bool operator()( const T& ref, pat::strbitset& ret ) override final {
    howfar_ = 0;
    bool failed = false;
    if( !initialized_ ) {
      throw cms::Exception("CutNotInitialized")
	<< "VersionedGsfElectronSelector not initialized!" << std::endl;
    }  
    for( unsigned i = 0; i < cuts_.size(); ++i ) {
      reco::CandidatePtr temp(ref);
      const bool result = (*cuts_[i])(temp);
      if( result || this->ignoreCut(cut_indices_[i]) ) {
	this->passCut(ret,cut_indices_[i]);
	if( !failed ) ++howfar_;
      } else {
	failed = true;
      }
    }
    this->setIgnored(ret);
    return (bool)ret;
  }
  
  virtual bool operator()(const T& ref) override final {
    this->retInternal_.set(false);
    this->operator()(ref, this->retInternal_);
    this->setIgnored(this->retInternal_);
    return (bool)this->retInternal_;
  }

  virtual bool operator()(const T& ref, edm::EventBase const& e, pat::strbitset& ret) override final {
    // setup isolation needs
    for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
      if( needs_event_content_[i] ) {
	CutApplicatorWithEventContentBase* needsEvent = 
	  static_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
	needsEvent->getEventContent(e); 
      }
    }
    return this->operator()(ref, ret);
  }

  using Selector<T>::operator();

  const unsigned char* md55Raw() const { return id_md5_; } 
  bool operator==(const VersionedSelector& other) const {
    constexpr unsigned length = MD5_DIGEST_LENGTH;
    return ( 0 == memcmp(id_md5_,other.id_md5_,length*sizeof(unsigned char)) );
  }
  const std::string& md5String() const { return md5_string_; }

  const std::string& name() const { return name_; }

  const unsigned howFarInCutFlow() const { return howfar_; }

  const size_t cutFlowSize() const { return cuts_.size(); } 

  void initialize(const edm::ParameterSet&);

  void setConsumes(edm::ConsumesCollector);

 protected:
  bool initialized_;
  std::vector<std::shared_ptr<candf::CandidateCut> > cuts_;
  std::vector<bool> needs_event_content_;
  std::vector<typename Selector<T>::index_type> cut_indices_;
  unsigned howfar_;

 private:  
  unsigned char id_md5_[MD5_DIGEST_LENGTH];
  std::string md5_string_,name_;  
};

template<class T>
void VersionedSelector<T>::
initialize( const edm::ParameterSet& conf ) {
  if(initialized_) {
    edm::LogWarning("VersionedPatElectronSelector")
      << "ID was already initialized!";
    return;
  }  
  const std::vector<edm::ParameterSet>& cutflow =
    conf.getParameterSetVector("cutFlow");  
  if( cutflow.size() == 0 ) {
    throw cms::Exception("InvalidCutFlow")
      << "You have supplied a null/empty cutflow to VersionedIDSelector,"
      << " please add content to the cuflow and try again.";
  }
  
  // this lets us keep track of cuts without knowing what they are :D
  std::vector<edm::ParameterSet>::const_iterator cbegin(cutflow.begin()),
    cend(cutflow.end());
  std::vector<edm::ParameterSet>::const_iterator icut = cbegin;
  std::map<std::string,unsigned> cut_counter;
  for( ; icut != cend; ++icut ) {  
    std::stringstream realname;
    const std::string& name = icut->getParameter<std::string>("cutName");
    if( !cut_counter.count(name) ) cut_counter[name] = 0;      
    realname << name << "_" << cut_counter[name];
    const bool needsContent = 
      icut->getParameter<bool>("needsAdditionalProducts");     
    const bool ignored = icut->getParameter<bool>("isIgnored");
    candf::CandidateCut* plugin = 
      CutApplicatorFactory::get()->create(name,*icut);
    if( plugin != nullptr ) {
      cuts_.push_back(std::shared_ptr<candf::CandidateCut>(plugin));
    } else {
      throw cms::Exception("BadPluginName")
	<< "The requested cut: " << name << " is not available!";
    }
    needs_event_content_.push_back(needsContent);

    const std::string therealname = realname.str();
    this->push_back(therealname);
    this->set(therealname);
    if(ignored) this->ignoreCut(therealname);
    cut_counter[name]++;
  }    

  //have to loop again to set cut indices after all are filled
  icut = cbegin;
  cut_counter.clear();
  for( ; icut != cend; ++icut ) {
    std::stringstream realname;
    const std::string& name = icut->getParameter<std::string>("cutName");    
    if( !cut_counter.count(name) ) cut_counter[name] = 0;      
    realname << name << "_" << cut_counter[name];
    cut_indices_.push_back(typename Selector<T>::index_type(&(this->bits_),realname.str()));    
    cut_counter[name]++;
  }
  
  initialized_ = true;
}

//#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
template<class T>
void VersionedSelector<T>::setConsumes(edm::ConsumesCollector cc) {
  for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
    if( needs_event_content_[i] ) {
      CutApplicatorWithEventContentBase* needsEvent = 
	static_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
      needsEvent->setConsumes(cc);
    }
  }
}
//#endif

#endif
