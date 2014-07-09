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
    memset(id_md5_,0,length*sizeof(unsigned char));
    std::string tracked, untracked;
    conf.toString(tracked); // get tracked PSet
    conf.allToString(untracked); // get untracked parts
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

 protected:
  std::vector<boost::shared_ptr<candf::CandidateCut> > cuts_;
  std::vector<bool> is_isolation_;

 private:
  unsigned char id_md5_[MD5_DIGEST_LENGTH];
  std::string md5_string_;
};

#endif
