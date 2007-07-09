#ifndef Common_IndirectVectorHolder_h
#define Common_IndirectVectorHolder_h
#include "DataFormats/Common/interface/BaseVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "DataFormats/Common/interface/IndirectHolder.h"

namespace edm {
  namespace reftobase {

    template <class T>
    class IndirectVectorHolder : public BaseVectorHolder<T> {
    public:
      typedef BaseVectorHolder<T>                base_type;
      typedef typename base_type::size_type      size_type;
      typedef typename base_type::element_type   element_type;
      typedef typename base_type::base_ref_type  base_ref_type;
      typedef typename base_type::const_iterator const_iterator;
      
      IndirectVectorHolder();
      IndirectVectorHolder( const IndirectVectorHolder & other);
      IndirectVectorHolder(boost::shared_ptr<RefVectorHolderBase> p);
      virtual ~IndirectVectorHolder();
      IndirectVectorHolder& operator= (IndirectVectorHolder const& rhs);
      void swap(IndirectVectorHolder& other);
      virtual BaseVectorHolder<T>* clone() const;
      virtual ProductID id() const;
      virtual EDProductGetter const* productGetter() const;
      virtual bool empty() const;
      virtual size_type size() const;
      virtual void clear();
      virtual base_ref_type const at(size_type idx) const;
      virtual std::auto_ptr<reftobase::RefVectorHolderBase> vectorHolder() const {
	return std::auto_ptr<reftobase::RefVectorHolderBase>( helper_->clone() );
      }
      virtual void push_back( const BaseHolder<T> * r ) {
	typedef IndirectHolder<T> holder_type;
	const holder_type * h = dynamic_cast<const holder_type *>( r );
	if( h == 0 )
	  throw edm::Exception( edm::errors::InvalidReference ) 
	    << "In IndirectHolder<T> trying to push_back wrong reference type";
	helper_->push_back( h->helper_ );
      }
    private:
      typedef typename base_type::const_iterator_imp const_iterator_imp;
      RefVectorHolderBase * helper_;

    public:
      struct const_iterator_imp_specific : public const_iterator_imp {
	typedef ptrdiff_t difference_type;
	const_iterator_imp_specific() { }
	explicit const_iterator_imp_specific( const typename RefVectorHolderBase::const_iterator & it ) : i ( it ) { }
	~const_iterator_imp_specific() { }
	const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific( i ); }
	void increase() { ++i; }
	void decrease() { --i; }
	void increase( difference_type d ) { i += d; }
	void decrease( difference_type d ) { i -= d; }
	bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
	bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
	void assign( const const_iterator_imp * o ) { i = dc( o ); }
	base_ref_type deref() const { 
	  return base_ref_type( * i );
	}
	difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
      private:
	const typename RefVectorHolderBase::const_iterator & dc( const const_iterator_imp * o ) const {
	  if ( o == 0 )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In IndirectVectorHolder trying to dereference a null pointer";
	  const const_iterator_imp_specific * oo = dynamic_cast<const const_iterator_imp_specific *>( o );
	  if ( oo == 0 ) 
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In IndirectVectorHolder trying to cast iterator to wrong type ";
	  return oo->i;
	}    
	typename RefVectorHolderBase::const_iterator i;
      };

      const_iterator begin() const { 
	return const_iterator( new const_iterator_imp_specific( helper_->begin() ) ); 
      }
      const_iterator end() const { 
	return const_iterator( new const_iterator_imp_specific( helper_->end() ) ); 
      }
    };

    template <class T>
    IndirectVectorHolder<T>::IndirectVectorHolder() : helper_( 0 ) { }

    template <class T>
    IndirectVectorHolder<T>::IndirectVectorHolder(boost::shared_ptr<RefVectorHolderBase> p) :
      helper_(p->clone()) { }

    template <class T>
    IndirectVectorHolder<T>::IndirectVectorHolder( const IndirectVectorHolder & other ) :
      helper_( other.helper_->clone() ) { }

    template <class T>
    IndirectVectorHolder<T>::~IndirectVectorHolder() { 
      delete helper_;
    }

    template <class T>
    inline IndirectVectorHolder<T>& 
    IndirectVectorHolder<T>::operator= (IndirectVectorHolder const& rhs) {
      IndirectVectorHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <class T>
    inline void IndirectVectorHolder<T>::swap(IndirectVectorHolder& other) {
      std::swap(helper_, other.helper_);
    }

    template <class T>
    BaseVectorHolder<T>* 
    IndirectVectorHolder<T>::clone() const {
      return new IndirectVectorHolder<T>(*this);
    }
    
    template <class T>
    ProductID
    IndirectVectorHolder<T>::id() const {
      return helper_->id();
    }

    template <class T>
    EDProductGetter const* IndirectVectorHolder<T>::productGetter() const {
      return helper_->productGetter();
    }

    template <class T>
    bool IndirectVectorHolder<T>::empty() const {
      return helper_->empty();
    }

    template <class T>
    typename IndirectVectorHolder<T>::size_type IndirectVectorHolder<T>::size() const {
      return helper_->size();
    }

    template <class T>
    void IndirectVectorHolder<T>::clear() {
      return helper_->clear();
    }

    template <class T>
    typename IndirectVectorHolder<T>::base_ref_type const IndirectVectorHolder<T>::at(size_type idx) const {
      return helper_->template getRef<T>( idx );
    }

  }
}

#endif
