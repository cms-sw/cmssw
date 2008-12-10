#ifndef DataFormats_Common_RefVectorHolder_h
#define DataFormats_Common_RefVectorHolder_h
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  namespace reftobase {
    class RefHolderBase;
    template <class REF> class RefHolder;

    template<typename REFV>
    class RefVectorHolder : public RefVectorHolderBase  {
    public:
      RefVectorHolder() : RefVectorHolderBase() { }
      RefVectorHolder(const REFV & refs) : RefVectorHolderBase(), refs_(refs) { }
      virtual ~RefVectorHolder() { }
      void swap(RefVectorHolder& other);
      RefVectorHolder& operator=(RefVectorHolder const& rhs);
      virtual bool empty() const;
      virtual size_type size() const;
      virtual void clear();
      virtual void push_back( const RefHolderBase * r );
      virtual void reserve( size_type n );
      virtual ProductID id() const;
      virtual EDProductGetter const* productGetter() const;
      virtual RefVectorHolder<REFV> * clone() const;
      virtual RefVectorHolder<REFV> * cloneEmpty() const;
      void setRefs( const REFV & refs );
      virtual void reallyFillView( const void *, const ProductID &, std::vector<void const*> & );

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
	boost::shared_ptr<RefHolderBase> deref() const;
	difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
      private:
	const typename REFV::const_iterator & dc( const const_iterator_imp * o ) const {
	  if ( o == 0 )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefVectorHolder trying to dereference a null pointer\n";
	  const const_iterator_imp_specific * oo = dynamic_cast<const const_iterator_imp_specific *>( o );
	  if ( oo == 0 ) 
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefVectorHolder trying to cast iterator to wrong type\n";
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
      virtual const void * product() const {
	return refs_.product();
      }

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const { return refs_.isAvailable(); }

    private:
      virtual boost::shared_ptr<reftobase::RefHolderBase> refBase( size_t idx ) const;
      REFV refs_;
    };
    
    //
    // implementations for RefVectorHolder<REFV>
    //

    template <typename REFV>
    inline
    void RefVectorHolder<REFV>::swap(RefVectorHolder<REFV>& other) {
      this->RefVectorHolderBase::swap(other);
      refs_.swap(other.refs_);
    }

    template <typename REFV>
    inline
    RefVectorHolder<REFV>& RefVectorHolder<REFV>::operator=(RefVectorHolder<REFV> const& rhs) {
      RefVectorHolder<REFV> temp(rhs);
      this->swap(temp);
      return *this;
    }

    template<typename REFV>
    inline
    bool RefVectorHolder<REFV>::empty() const {
      return refs_.empty();
    }

    template<typename REFV>
    inline
    typename RefVectorHolder<REFV>::size_type RefVectorHolder<REFV>::size() const {
      return refs_.size();
    }
    
    template<typename REFV>
    inline
    void RefVectorHolder<REFV>::clear() {
      return refs_.clear();
    }

    template<typename REFV>
    inline
    void RefVectorHolder<REFV>::reserve( size_type n ) {
      typename REFV::size_type s = n;
      refs_.reserve( s );
    }

    template<typename REFV>
    inline
    ProductID RefVectorHolder<REFV>::id() const {
      return refs_.id();
    }

    template<typename REFV>
    inline
    EDProductGetter const* RefVectorHolder<REFV>::productGetter() const {
      return refs_.productGetter();
    }

    template<typename REFV>
    inline
    RefVectorHolder<REFV> * RefVectorHolder<REFV>::clone() const {
      return new RefVectorHolder<REFV>( * this );
    }

    template<typename REFV>
    inline
    RefVectorHolder<REFV> * RefVectorHolder<REFV>::cloneEmpty() const {
      return new RefVectorHolder<REFV>( id() );
    }

    template<typename REFV>
    inline
    void RefVectorHolder<REFV>::setRefs( const REFV & refs ) {
      refs_ = refs;
    }
    
    // Free swap function
    template <typename REFV>
    inline
    void
    swap(RefVectorHolder<REFV>& lhs, RefVectorHolder<REFV>& rhs) {
      lhs.swap(rhs);
    }
  }
}

#include "DataFormats/Common/interface/RefHolder.h"

namespace edm {
  namespace reftobase {
    
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

    template <class REFV>
    boost::shared_ptr<RefHolderBase>  
    RefVectorHolder<REFV>::refBase(size_t idx) const {
      return boost::shared_ptr<RefHolderBase>( new RefHolder<typename REFV::value_type>( refs_[idx] ) );
    }

    template<typename REFV>
    boost::shared_ptr<RefHolderBase> RefVectorHolder<REFV>::const_iterator_imp_specific::deref() const { 
      return boost::shared_ptr<RefHolderBase>( new RefHolder<typename REFV::value_type>( * i ) );
    }

  }
}

#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/traits.h"
#include "boost/mpl/if.hpp"

namespace edm {
  namespace reftobase {
    template<typename REFV>
    struct RefVectorHolderNoFillView {
      static void reallyFillView(RefVectorHolder<REFV>&, const void *, const ProductID &, std::vector<void const*> & ) {
	throw Exception(errors::ProductDoesNotSupportViews)
	  << "The product type " 
	  << typeid(typename REFV::collection_type).name()
	  << "\ndoes not support Views\n";
      }
    };

    template<typename REFV>
    struct RefVectorHolderDoFillView {
      static void reallyFillView(RefVectorHolder<REFV>& rvh, const void * prod, const ProductID & id , std::vector<void const*> & pointers ) {
	typedef typename REFV::collection_type collection;
	const collection * product = static_cast<const collection *>( prod );
	detail::reallyFillView( * product, id, pointers, rvh );
      }
    };    

    template<typename REFV>
    void RefVectorHolder<REFV>::reallyFillView( const void * prod, const ProductID & id , std::vector<void const*> & pointers ) {
      typedef 
	typename boost::mpl::if_c<has_fillView<typename REFV::collection_type>::value,
	RefVectorHolderDoFillView<REFV>,
	RefVectorHolderNoFillView<REFV> >::type maybe_filler;      
      maybe_filler::reallyFillView( *this, prod, id, pointers );
    }
  }
}

#endif
