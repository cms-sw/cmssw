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

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif

// because we need to be able to validate the ID
#include <openssl/md5.h>

#include <boost/shared_ptr.hpp>

namespace candf = candidate_functions;

template<class T>
class VersionedSelector : public Selector<T> {
 public:
 VersionedSelector() : Selector<T>() {}

 VersionedSelector(const edm::ParameterSet& conf) : 
  Selector<T>() { 
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
  }
  
  const unsigned char* md55Raw() const { return id_md5_; } 
  bool isSameID(const VersionedSelector& other) const {
    constexpr unsigned length = MD5_DIGEST_LENGTH;
    return ( 0 == memcmp(id_md5_,other.getMD5(),length*sizeof(unsigned char)) );
  }
  const std::string& md5String() const { return md5_string_; }

  const std::string& name() const { return name_; }

  const unsigned howFarInCutFlow() const { return howfar_; }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  void setConsumes(edm::ConsumesCollector);
#endif

 protected:
  std::vector<boost::shared_ptr<candf::CandidateCut> > cuts_;
  std::vector<bool> is_isolation_;
  std::vector<typename Selector<T>::index_type> cut_indices_;
  unsigned howfar_;

 private:
  unsigned char id_md5_[MD5_DIGEST_LENGTH];
   std::string md5_string_,name_;
};

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "PhysicsTools/SelectorUtils/interface/IsolationCutApplicatorBase.h"
template<class T>
void VersionedSelector<T>::setConsumes(edm::ConsumesCollector cc) {
  for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
    if( is_isolation_[i] ) {
      IsolationCutApplicatorBase* asIso = 
	static_cast<IsolationCutApplicatorBase*>(cuts_[i].get());
      asIso->setConsumes(cc);
    }
  }
}
#endif

#endif
