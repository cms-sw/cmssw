#ifndef Common_RefToBaseVector_h
#define Common_RefToBaseVector_h
/**\class RefToBaseVector
 *
 * \author Luca Lista, INFN
 *
 */

#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {
  namespace reftobase {
    template <class T>
    class BaseVectorHolder {
    public:
      BaseVectorHolder() {}
      virtual ~BaseVectorHolder() {}
      virtual BaseVectorHolder<T>* clone() const = 0;
      virtual RefToBase<T> const at(unsigned int idx) const = 0;
      virtual bool empty() const = 0;
      virtual unsigned int size() const = 0;
      virtual unsigned int capacity() const = 0;
      virtual void reserve(unsigned int n) = 0;
      virtual void clear() = 0;
      virtual ProductID id() const = 0;
    };

    template <class T, class TRefVector>
    class VectorHolder : public BaseVectorHolder<T> {
    public:
      VectorHolder() {}
      explicit VectorHolder(const TRefVector& iRefVector) : refVector_(iRefVector) {}
      virtual ~VectorHolder() {}
      virtual BaseVectorHolder<T>* clone() const { return new VectorHolder<T,TRefVector>(*this); }
      RefToBase<T> const at(unsigned int idx) const { return RefToBase<T>( refVector_.at( idx ) ); }
      bool empty() const { return refVector_.empty(); }
      unsigned int size() const { return refVector_.size(); }
      unsigned int capacity() const { return refVector_.capacity(); }
      void reserve(unsigned int n) { refVector_.reserve(n); }
      void clear() { refVector_.clear(); }
      ProductID id() const { return refVector_.id(); } 
   private:
      TRefVector refVector_;
    };
  }
  
  template <class T>
  class RefToBaseVector {
  public:
    typedef RefToBase<T> value_type;
    typedef T member_type;
    typedef unsigned int size_type;

    RefToBaseVector() : holder_(0) { }
    template <class TRefVector>
    explicit RefToBaseVector(const TRefVector& iRef) : holder_(new reftobase::VectorHolder<T,TRefVector>(iRef)) { }
    RefToBaseVector(const RefToBaseVector<T>& iOther): 
      holder_((0==iOther.holder_) ? static_cast<reftobase::BaseVectorHolder<T>*>(0) : iOther.holder_->clone()) {
    }
    const RefToBaseVector& operator=(const RefToBaseVector<T>& iRHS) {
      RefToBaseVector<T> temp(iRHS);
      this->swap(temp);
      return *this;
    }
    ~RefToBaseVector() { delete holder_; }

    value_type const at(size_type idx) const { return holder_->at( idx ); }    
    value_type const operator[](size_type idx) const { return at( idx ); }
    bool empty() const { return holder_->empty(); }
    size_type size() const { return holder_->size(); }
    size_type capacity() const { return holder_->capacity(); }
    void reserve(unsigned int n) { holder_->reserve(n); }
    void clear() { holder_->clear(); }
    ProductID id() const {return holder_->id();}

  private:
    reftobase::BaseVectorHolder<T>* holder_;
  };
  
  // Free swap function
  template <class T>
  inline
  void
  swap(RefToBaseVector<T>& a, RefToBaseVector<T>& b) {
    a.swap(b);
  }

}

#endif
