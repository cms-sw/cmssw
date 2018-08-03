#ifndef DETECTOR_DESCRIPTION_CORE_DDBASE_H
#define DETECTOR_DESCRIPTION_CORE_DDBASE_H

#include <utility>
#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/rep_type.h"

template <class N, class C>
  class DDBase
{
 public:
  template <class D>
    class iterator
    {
    public:
      //! C is, for example, a DDLogicalPart or a DDMaterial or a DDSolid ...
      using value_type = D;

      explicit iterator( const typename DDI::Store<N,C>::iterator it ) : it_( it ) {}
        
      iterator() : it_( StoreT::instance().begin()) {}
  
      value_type& operator*() const {
	d_.prep_ = it_->second;
	return d_;
      }
  
      value_type* operator->() const {
	d_.prep_ = it_->second;
	return &d_;
      }
  
      bool operator==( const iterator & i ) const {
	return i.it_ == it_;
      }
    
      bool operator!=( const iterator & i ) const {
	return i.it_ != it_;
      }
      
      bool operator<( const iterator & i ) const {
	return it_ < i.it_;
      }
    
      bool operator>( const iterator & i ) const {
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

  using StoreT = DDI::Singleton< DDI::Store< N, C > >;
  using def_type = std::pair< const N*, bool >;
  static auto end()   { return StoreT::instance().end(); }
  static auto begin() { return StoreT::instance().begin(); }

  DDBase() : prep_(nullptr) { }
  virtual ~DDBase() { /*never do this here: if (prep_) delete prep_;*/ }
  
  const N & name() const { return prep_->name(); }
  
  const N & ddname() const { return prep_->name(); }
  
  std::string toString() const { return prep_->name().fullname(); }

  const typename DDI::rep_traits< N, C >::reference rep() const {
    return *( prep_->second );
  }
  
  typename DDI::rep_traits< N, C >::reference rep() {
    return *( prep_->second );
  }
    
  const typename DDI::rep_traits< N, C >::reference val() const {
    if( !isValid()) throw cms::Exception( "DDException" ) << "undefined: " << name(); 
    return rep();
  }  

  const typename DDI::rep_traits<N,C>::reference val() {
    if( !isValid()) throw cms::Exception( "DDException" ) << "undefined: " << name(); 
    return rep();
  }
  
  bool operator==( const DDBase & b ) const { return prep_ == b.prep_; }
  // true, if registered or defined
  operator bool() const { return isValid(); }
  bool operator<( const DDBase & b ) const { return prep_ < b.prep_; }
  bool operator>( const DDBase & b ) const { return prep_ > b.prep_; }
  
  // ( name*, true ) if defined
  // ( name*, false ) if registered but not defined
  // ( 0, false ) if not there at all
  def_type isDefined() const 
  {
    return prep_ ?
      std::make_pair( &( prep_->name()), bool( prep_->second ))
      :
      std::make_pair(( const N* )nullptr, false );  
  }
  
  //! true, if the wrapped pointer is valid
  bool isValid() const
  {
    return prep_ ? bool( prep_->second )
      : false;
  }
  void create( const N& name, C vals ) {
    prep_ = StoreT::instance().create( name, vals );
  }
  void create( const N& name ) {
    prep_ = StoreT::instance().create( name );
  }

 private:
  DDI::rep_type<N, C>* prep_; 
};

#endif
