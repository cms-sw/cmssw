#ifndef PhysicsTools_SelectorUtils_VersionedSelector_h
#define PhysicsTools_SelectorUtils_VersionedSelector_h

/**
  \class    VersionedSelector VersionedSelector.h "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
  \brief    cut-flow versioning info in the event provenance

  class template to implement versioning for IDs that's available in the 
  event provenance or available by hash-code in the event record

  \author Lindsey Gray
*/

#if ( !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__) ) || defined(__ROOTCLING__)

#define REGULAR_CPLUSPLUS 1
#define CINT_GUARD(CODE) CODE
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <memory>
#define SHARED_PTR(T) std::shared_ptr<T>

#else

#define CINT_GUARD(CODE)
#include <boost/shared_ptr.hpp>
#define SHARED_PTR(T) boost::shared_ptr<T>

#endif

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/SelectorUtils/interface/CandidateCut.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

// because we need to be able to validate the ID
#include <openssl/md5.h>

namespace candf = candidate_functions;

namespace vid {
  class CutFlowResult;
}

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
	<< " in the cutflow ParameterSet!";
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
  
  virtual bool operator()( const T& ref, pat::strbitset& ret ) CINT_GUARD(override final) {
    howfar_ = 0;
    bitmap_ = 0;
    values_.clear();
    bool failed = false;
    if( !initialized_ ) {
      throw cms::Exception("CutNotInitialized")
	<< "VersionedGsfElectronSelector not initialized!" << std::endl;
    }  
    for( unsigned i = 0; i < cuts_.size(); ++i ) {
      reco::CandidatePtr temp(ref);
      const bool result = (*cuts_[i])(temp);
      values_.push_back(cuts_[i]->value(temp));
      if( result || this->ignoreCut(cut_indices_[i]) ) {
	this->passCut(ret,cut_indices_[i]);
        bitmap_ |= 1<<i;
	if( !failed ) ++howfar_;
      } else {
	failed = true;
      }
    }
    this->setIgnored(ret);
    return (bool)ret;
  }
  
  virtual bool operator()(const T& ref, edm::EventBase const& e, pat::strbitset& ret) CINT_GUARD(override final) {
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
  
  //repeat the other operator() we left out here
  //in the base class here so they are exposed to ROOT

  /* VID BY VALUE */
  bool operator()( typename T::value_type const & t ) {
    const T temp(&t,0); // assuming T is edm::Ptr
    return this->operator()(temp);
  }

  bool operator()( typename T::value_type const & t, edm::EventBase const & e) {
    const T temp(&t,0);
    return this->operator()(temp,e);
  }
  
  virtual bool operator()( T const & t ) CINT_GUARD(override final) {
    this->retInternal_.set(false);
    this->operator()(t, this->retInternal_);
    this->setIgnored(this->retInternal_);
    return (bool)this->retInternal_;
  }
  
  virtual bool operator()( T const & t, edm::EventBase const & e) CINT_GUARD(override final) {
    this->retInternal_.set(false);
    this->operator()(t, e, this->retInternal_);
    this->setIgnored(this->retInternal_);
    return (bool)this->retInternal_;
  }

  const unsigned char* md55Raw() const { return id_md5_; } 
  bool operator==(const VersionedSelector& other) const {
    constexpr unsigned length = MD5_DIGEST_LENGTH;
    return ( 0 == memcmp(id_md5_,other.id_md5_,length*sizeof(unsigned char)) );
  }
  const std::string& md5String() const { return md5_string_; }

  const std::string& name() const { return name_; }

  const unsigned howFarInCutFlow() const { return howfar_; }
  
  const unsigned bitMap() const { return bitmap_; }

  const size_t cutFlowSize() const { return cuts_.size(); } 

  vid::CutFlowResult cutFlowResult() const;

  void initialize(const edm::ParameterSet&);

  CINT_GUARD(void setConsumes(edm::ConsumesCollector));

 protected:
  bool initialized_;
  std::vector<SHARED_PTR(candf::CandidateCut) > cuts_;
  std::vector<bool> needs_event_content_;
  std::vector<typename Selector<T>::index_type> cut_indices_;
  unsigned howfar_, bitmap_;
  std::vector<double> values_;

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
    const std::string& cname = icut->getParameter<std::string>("cutName");
    const bool needsContent = 
      icut->getParameter<bool>("needsAdditionalProducts");     
    const bool ignored = icut->getParameter<bool>("isIgnored");
    candf::CandidateCut* plugin = nullptr;
    CINT_GUARD(plugin = CutApplicatorFactory::get()->create(cname,*icut));
    if( plugin != nullptr ) {
      cuts_.push_back(SHARED_PTR(candf::CandidateCut)(plugin));
    } else {
      throw cms::Exception("BadPluginName")
	<< "The requested cut: " << cname << " is not available!";
    }
    needs_event_content_.push_back(needsContent);
    const std::string& name = plugin->name();
    std::stringstream realname;    
    if( !cut_counter.count(name) ) cut_counter[name] = 0;      
    realname << name << "_" << cut_counter[name];
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
    const std::string& name = cuts_[std::distance(cbegin,icut)]->name();    
    if( !cut_counter.count(name) ) cut_counter[name] = 0;      
    realname << name << "_" << cut_counter[name];
    cut_indices_.push_back(typename Selector<T>::index_type(&(this->bits_),realname.str()));    
    cut_counter[name]++;
  }
  
  initialized_ = true;
}



#ifdef REGULAR_CPLUSPLUS
#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
template<class T> 
vid::CutFlowResult VersionedSelector<T>::cutFlowResult() const {
  std::map<std::string,unsigned> names_to_index;
  std::map<std::string,unsigned> cut_counter;
  for( unsigned idx = 0; idx < cuts_.size(); ++idx ) {
    const std::string& name = cuts_[idx]->name();
    if( !cut_counter.count(name) ) cut_counter[name] = 0;  
    std::stringstream realname;
    realname << name << "_" << cut_counter[name];
    names_to_index.emplace(realname.str(),idx);
    cut_counter[name]++;
  }
  return vid::CutFlowResult(name_,md5_string_,names_to_index,values_,bitmap_);
}

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
template<class T>
void VersionedSelector<T>::setConsumes(edm::ConsumesCollector cc) {
  for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
    if( needs_event_content_[i] ) {
      CutApplicatorWithEventContentBase* needsEvent = 
	dynamic_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
      if( nullptr != needsEvent ) {
        needsEvent->setConsumes(cc);
      } else {
        throw cms::Exception("InvalidCutConfiguration")
          << "Cut: " << ((CutApplicatorBase*)cuts_[i].get())->name() 
          << " configured to consume event products but does not "
          << " inherit from CutApplicatorWithEventContenBase "
          << " please correct either your python or C++!";
      }
    }
  }
}
#endif

#endif
