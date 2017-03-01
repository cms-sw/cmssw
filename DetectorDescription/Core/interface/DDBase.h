#ifndef DDCore_DDBase_h
#define DDCore_DDBase_h

#include <utility>
#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/Base/interface/Store.h"
#include "DetectorDescription/Base/interface/rep_type.h"

template <class N, class C>
class DDBase
{
public:
  template <class D>
  class iterator
  {
  public:
    //! C is, for example, a DDLogicalPart or a DDMaterial or a DDSolid ....
    typedef D value_type;

    explicit iterator( const typename DDI::Store<N,C>::iterator it) : it_(it) { }
        
    iterator() : it_(StoreT::instance().begin()) {  }
  
    value_type& operator*() const {
      d_.prep_ = it_->second;
      return d_;
    }
  
    value_type* operator->() const {
      d_.prep_ = it_->second;
      return &d_;
    }
  
    bool operator==(const iterator & i) const {
      return i.it_ == it_;
    }
    
    bool operator!=(const iterator & i) const {
      return i.it_ != it_;
    }
    
    bool operator<(const iterator & i) const {
      return it_ < i.it_;
    }
    
    bool operator>(const iterator & i) const {
      return it_ > i.it_;
    }
    
    void operator++() {
      ++it_;
    }
    
    void end() const {
      it_ = StoreT::instance().end();
    }  
        
     
  private:
    mutable typename DDI::Store<N,C>::iterator it_;
    mutable D d_;
  };
   
public:
  static auto end()   { return StoreT::instance().end(); }
  static auto begin() { return StoreT::instance().begin(); }
  // dangerous stuff!!
  static void clear() { StoreT::instance().clear(); }
  static size_t size() { return StoreT::instance().size(); }
  typedef DDI::Singleton<DDI::Store<N,C> > StoreT;
  typedef C pimpl_type;
  typedef DDI::rep_type<N,pimpl_type>* prep_type;
  typedef std::pair<const N*,bool> def_type;
  
  DDBase() : prep_(0) { }
  virtual ~DDBase() { /*never do this here: if (prep_) delete prep_;*/ }
      
  const N & name() const { return prep_->name(); }
  
  const N & ddname() const { return prep_->name(); }
  
  std::string toString() const { return prep_->name().fullname(); }

  const typename DDI::rep_traits<N,C>::reference rep() const 
    { return *(prep_->second); }
  
  typename DDI::rep_traits<N,C>::reference rep() 
    { return *(prep_->second); }
    
  const typename DDI::rep_traits<N,C>::reference val() const
   { if (!isValid()) throw cms::Exception("DDException") << "undefined: " << name(); 
      return rep();
    };  

  const typename DDI::rep_traits<N,C>::reference val() 
   { if (!isValid()) throw cms::Exception("DDException") << "undefined: " << name(); 
      return rep();
    };  
  
  bool operator==(const DDBase & b) const { return prep_ == b.prep_; }
  // true, if registered or defined
  operator bool() const { return isValid(); }
  
  bool operator<(const DDBase & b) const { return prep_ < b.prep_; }
  bool operator>(const DDBase & b) const { return prep_ > b.prep_; }
  
  // (name*,true) if defined
  // (name*,false) if registered but not defined
  // (0,false) if not there at all
  def_type isDefined() const 
    {
      return prep_ ?
	std::make_pair(&(prep_->name()), bool(prep_->second))
	:
	std::make_pair((const N *)0,false);  
    } 
   
  //! true, if the  wrapped pointer is valid
  bool isValid() const
    {
      return prep_ ? bool(prep_->second)
	: false;
    } 
  
protected: 
  prep_type prep_; 
};

#endif
