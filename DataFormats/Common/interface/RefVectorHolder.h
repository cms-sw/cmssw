#ifndef Common_RefVectorHolder_h
#define Common_RefVectorHolder_h
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace reftobase {
    template<typename REFV>
    class RefVectorHolder : public RefVectorHolderBase  {
    public:
      RefVectorHolder() { }
      RefVectorHolder( const REFV & refs ) : refs_( refs ) { }
      virtual ~RefVectorHolder() { }
      virtual bool empty() const;
      virtual size_type size() const;
      virtual void clear();
      virtual void push_back( const RefHolderBase * r );
      virtual void reserve( size_type n );
      virtual ProductID id() const;
      virtual EDProductGetter const* productGetter() const;
      virtual RefVectorHolder<REFV> * clone() const;
      void setRefs( const REFV & refs );

    private:
      typedef typename RefVectorHolderBase::const_iterator_imp const_iterator_imp;

    public:      
      struct const_iterator_imp_specific : public const_iterator_imp {
	typedef ptrdiff_t difference_type;
	const_iterator_imp_specific() { }
	explicit const_iterator_imp_specific( const typename REFV::const_iterator & it ) : i ( it ) { }
	~const_iterator_imp_specific() { }
	const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific( i ); }
	void increase() { ++i; }
	void decrease() { --i; }
	void increase( difference_type d ) { i += d; }
	void decrease( difference_type d ) { i -= d; }
	bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
	bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
	void assign( const const_iterator_imp * o ) { i = dc( o ); }
	boost::shared_ptr<RefHolderBase> deref() const { 
	  return boost::shared_ptr<RefHolderBase>( new RefHolder<typename REFV::value_type>( * i ) );
	}
	difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
      private:
	const typename REFV::const_iterator & dc( const const_iterator_imp * o ) const {
	  if ( o == 0 )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefVectorHolder trying to dereference a null pointer";
	  const const_iterator_imp_specific * oo = dynamic_cast<const const_iterator_imp_specific *>( o );
	  if ( oo == 0 ) 
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefVectorHolder trying to cast iterator to wrong type ";
	  return oo->i;
	} 
	typename REFV::const_iterator i;
      };
      
      typedef typename RefVectorHolderBase::const_iterator const_iterator;
      
      const_iterator begin() const { 
	return const_iterator( new const_iterator_imp_specific( refs_.begin() ) ); 
      }
      const_iterator end() const { 
	return const_iterator( new const_iterator_imp_specific( refs_.end() ) ); 
      }

    private:
      virtual boost::shared_ptr<reftobase::RefHolderBase> refBase( size_t idx ) const;
      REFV refs_;
    };
    
    //
    // implementations for RefVectorHolder<REFV>
    //

    template<typename REFV>
    bool RefVectorHolder<REFV>::empty() const {
      return refs_.empty();
    }

    template<typename REFV>
    typename RefVectorHolder<REFV>::size_type RefVectorHolder<REFV>::size() const {
      return refs_.size();
    }
    
    template<typename REFV>
    void RefVectorHolder<REFV>::clear() {
      return refs_.clear();
    }
    
    template<typename REFV>
    void RefVectorHolder<REFV>::push_back( const RefHolderBase * h ) {
      typedef typename REFV::value_type REF;
      const RefHolder<REF> * rh = dynamic_cast<const RefHolder<REF> *>( h );
      if( rh == 0 )
	throw edm::Exception(errors::InvalidReference)
	  << "RefVectorHolder: attempting to cast a RefHolderBase "
	  << "to an invalid type.\nExpected: "
	  << typeid( REF ).name() << "\n";
      refs_.push_back( rh->getRef() );
    }

    template<typename REFV>
    void RefVectorHolder<REFV>::reserve( size_type n ) {
      typename REFV::size_type s = n;
      refs_.reserve( s );
    }

    template<typename REFV>
    ProductID RefVectorHolder<REFV>::id() const {
      return refs_.id();
    }

    template<typename REFV>
    EDProductGetter const* RefVectorHolder<REFV>::productGetter() const {
      return refs_.productGetter();
    }

    template<typename REFV>
    RefVectorHolder<REFV> * RefVectorHolder<REFV>::clone() const {
      return new RefVectorHolder<REFV>( * this );
    }

    template<typename REFV>
    void RefVectorHolder<REFV>::setRefs( const REFV & refs ) {
      refs_ = refs;
    }
    
    template <class REFV>
    boost::shared_ptr<RefHolderBase>  
    RefVectorHolder<REFV>::refBase(size_t idx) const {
      return boost::shared_ptr<RefHolderBase>( new RefHolder<typename REFV::value_type>( refs_[idx] ) );
    }


  }
}

#endif
