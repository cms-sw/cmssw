#include<memory>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace extstd {

  /* modify unique_ptr behaviour adding a "cloning" copy-constructor and assignment-operator
   */
  template<typename T>
  struct clone_ptr : public std::unique_ptr<T> {
    
    template<typename... Args>
    explicit clone_ptr(Args&& ... args)  noexcept : std::unique_ptr<T>(std::forward<Args>(args)...){}
    
    clone_ptr(clone_ptr const & rh) : std::unique_ptr<T>(rh? rh->clone() : nullptr){}
    clone_ptr(clone_ptr && rh) noexcept : std::unique_ptr<T>(std::move(rh)) {}
    
    clone_ptr & operator=(clone_ptr const & rh) {
      if (&rh!=this) this->reset(rh? rh->clone() : nullptr);
      return *this;
    }
    clone_ptr & operator=(clone_ptr && rh) noexcept {
      if (&rh!=this) std::unique_ptr<T>::operator=(std::move(rh));
      return *this;
    }
    
    
    template<typename U>   
    clone_ptr(clone_ptr<U> const & rh) : std::unique_ptr<T>(rh ? rh->clone() : nullptr){}
    template<typename U>
    clone_ptr(clone_ptr<U> && rh)  noexcept : std::unique_ptr<T>(std::move(rh)) {}
    
    template<typename U>
    clone_ptr & operator=(clone_ptr<U> const & rh) {
      if (&rh!=this) this->reset(rh? rh->clone() : nullptr);
      return *this;
    }
    template<typename U>
    clone_ptr & operator=(clone_ptr<U> && rh) noexcept {
      if (&rh!=this) std::unique_ptr<T>::operator=(std::move(rh));
      return *this;
    }
    
  };

}


