#ifndef Common_RefToBaseVector_h
#define Common_RefToBaseVector_h
/**\class RefToBaseVector
 *
 * \author Luca Lista, INFN
 *
 */

#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace reftobase {
    template <class T>
    class BaseVectorHolder {
    public:
      typedef size_t       size_type;
      typedef T            element_type;
      typedef RefToBase<T> base_ref_type;
      BaseVectorHolder() {}
      virtual ~BaseVectorHolder() {}
      virtual BaseVectorHolder* clone() const = 0;
      virtual base_ref_type const at(size_type idx) const = 0;
      virtual bool empty() const = 0;
      
      virtual size_type size() const = 0;
      //virtual size_type capacity() const = 0;
      //virtual void reserve(size_type n) = 0;
      virtual void clear() = 0;
      virtual ProductID id() const = 0;

      // the following structure is public 
      // to allow reflex dictionary to compile
      //    protected:
      struct const_iterator_imp {
	typedef ptrdiff_t difference_type;
	const_iterator_imp() { } 
	virtual ~const_iterator_imp() { }
	virtual const_iterator_imp * clone() const = 0;
	virtual void increase() = 0;
	virtual void decrease() = 0;
	virtual void increase( difference_type d ) = 0;
	virtual void decrease( difference_type d ) = 0;
	virtual bool equal_to( const const_iterator_imp * ) const = 0;
	virtual bool less_than( const const_iterator_imp * ) const = 0;
	virtual void assign( const const_iterator_imp * ) = 0;
	virtual const T & deref() const = 0;
	virtual difference_type difference( const const_iterator_imp * ) const = 0;
      };
    public:
      struct const_iterator : public std::iterator <std::random_access_iterator_tag, RefToBase<T> >{
	typedef T value_type;
	typedef T * pointer;
	typedef T & reference;
	typedef std::ptrdiff_t difference_type;
	const_iterator() : i( 0 ) { }
	const_iterator( const_iterator_imp * it ) : i( it ) { }
	const_iterator( const const_iterator & it ) : i( it.isValid() ? it.i->clone() : 0 ) { }
	~const_iterator() { delete i; }
	const_iterator & operator=( const const_iterator & it ) { 
	  if ( isInvalid() ) i = it.i;
	  else i->assign( it.i ); 
	  return *this; 
	}
	const_iterator& operator++() { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to increment an inavlid RefToBaseVector<T>::const_iterator";
	  i->increase(); 
	  return *this; 
	}
	const_iterator operator++( int ) { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to postincrement an inavlid RefToBaseVector<T>::const_iterator";
	  const_iterator ci = *this; 
	  i->increase(); 
	  return ci; 
	}
	const_iterator& operator--() { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to decrement an inavlid RefToBaseVector<T>::const_iterator";
	  i->decrease(); 
	  return *this; 
	}
	const_iterator operator--( int ) { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to postdecrement an inavlid RefToBaseVector<T>::const_iterator";
	  const_iterator ci = *this; 
	  i->decrease(); 
	  return ci; 
	}
	difference_type operator-( const const_iterator & o ) const { 
	  if ( isInvalid() && o.isInvalid() ) return 0;
	  if ( isInvalid() || o.isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to compute difference with an inavlid RefToBaseVector<T>::const_iterator";
	  return i->difference( o.i ); 
	}
	const_iterator operator+( difference_type n ) const { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to compute sum with an inavlid RefToBaseVector<T>::const_iterator";
	  const_iterator_imp * ii = i->clone(); 
	  ii->increase( n );
	  return const_iterator( ii ); 
	}
	const_iterator operator-( difference_type n ) const { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to compute difference with an inavlid RefToBaseVector<T>::const_iterator";
	  const_iterator_imp * ii = i->clone();
	  ii->decrease( n );
	  return const_iterator( ii ); 
	}
	bool operator<( const const_iterator & o ) const { 
	  if ( isInvalid() && o.isInvalid() ) return false;
	  if ( isInvalid() || o.isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to compute < operator with an inavlid RefToBaseVector<T>::const_iterator";
	  return i->less_than( o.i ); 
	}
	bool operator==( const const_iterator& ci ) const { 
	  if ( isInvalid() && ci.isInvalid() ) return true;
	  if ( isInvalid() || ci.isInvalid() ) return false;
	  return i->equal_to( ci.i ); 
	}
	bool operator!=( const const_iterator& ci ) const { 
	  if ( isInvalid() && ci.isInvalid() ) return false;
	  if ( isInvalid() || ci.isInvalid() ) return true;
	  return ! i->equal_to( ci.i ); 
	}
	const T & operator * () const { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to dereference an inavlid RefToBaseVector<T>::const_iterator";
	  return i->deref(); 
	}
	const T * operator->() const { return & ( operator*() ); }
	const_iterator & operator +=( difference_type d ) { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to increment an inavlid RefToBaseVector<T>::const_iterator";
	  i->increase( d ); 
	  return *this; 
	}
	const_iterator & operator -=( difference_type d ) { 
	  if ( isInvalid() )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "Trying to decrement an inavlid RefToBaseVector<T>::const_iterator";
	  i->decrease( d ); 
	  return *this; 
	}
	bool isValid() const { return i != 0; }
	bool isInvalid() const { return i == 0; }
      private:
	const_iterator_imp * i;
      };

      virtual const_iterator begin() const = 0;
      virtual const_iterator end() const = 0;
    };

    template <class T, class TRefVector>
    class VectorHolder : public BaseVectorHolder<T> {
    public:
      typedef BaseVectorHolder<T>                base_type;
      typedef typename base_type::size_type      size_type;
      typedef typename base_type::element_type   element_type;
      typedef typename base_type::base_ref_type  base_ref_type;
      typedef typename base_type::const_iterator const_iterator;
      typedef TRefVector                         ref_vector_type;

      VectorHolder() {}
      explicit VectorHolder(const ref_vector_type& iRefVector) : refVector_(iRefVector) {}
      virtual ~VectorHolder() {}
      virtual base_type* clone() const { return new VectorHolder(*this); }
      base_ref_type const at(size_type idx) const { return base_ref_type( refVector_.at( idx ) ); }
      bool empty() const { return refVector_.empty(); }
      size_type size() const { return refVector_.size(); }
      //size_type capacity() const { return refVector_.capacity(); }
      //void reserve(size_type n) { refVector_.reserve(n); }
      void clear() { refVector_.clear(); }
      ProductID id() const { return refVector_.id(); } 

      const_iterator begin() const { 
	return const_iterator( new const_iterator_imp_specific( refVector_.begin() ) ); 
      }
      const_iterator end() const { 
	return const_iterator( new const_iterator_imp_specific( refVector_.end() ) ); 
      }

    private:
      typedef typename base_type::const_iterator_imp const_iterator_imp;

      ref_vector_type refVector_;

      // the following structure is public 
      // to allow reflex dictionary to compile
    public:
      struct const_iterator_imp_specific : public const_iterator_imp {
	typedef ptrdiff_t difference_type;
	const_iterator_imp_specific() { }
	explicit const_iterator_imp_specific( const typename TRefVector::const_iterator & it ) : i ( it ) { }
	~const_iterator_imp_specific() { }
	const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific( i ); }
	void increase() { ++i; }
	void decrease() { --i; }
	void increase( difference_type d ) { i += d; }
	void decrease( difference_type d ) { i -= d; }
	bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
	bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
	void assign( const const_iterator_imp * o ) { i = dc( o ); }
	const T & deref() const { return * * i; }
	difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
      private:
	const typename ref_vector_type::const_iterator & dc( const const_iterator_imp * o ) const {
	  if ( o == 0 )
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefToBaseVector<T> trying to dereference a null pointer";
	  const const_iterator_imp_specific * oo = dynamic_cast<const const_iterator_imp_specific *>( o );
	  if ( oo == 0 ) 
	    throw edm::Exception( edm::errors::InvalidReference ) 
	      << "In RefToBaseVector<T> trying to cast iterator to wrong type ";
	  return oo->i;
	}    
	typename ref_vector_type::const_iterator i;
      };
    };

  }

  //--------------------------------------------------------------------
  //
  // Class template RefToBaseVector<T>
  //
  //--------------------------------------------------------------------

  /// RefToBaseVector<T> provides ... ?

  template <class T>
  class RefToBaseVector {
  public:
    typedef RefToBase<T>                         value_type;
    typedef T                                    member_type;
    typedef reftobase::BaseVectorHolder<T>       holder_type;
    typedef typename holder_type::size_type      size_type;
    typedef typename holder_type::const_iterator const_iterator;

    RefToBaseVector();
    RefToBaseVector(RefToBaseVector const& iOther);
    template <class TRefVector> explicit RefToBaseVector(TRefVector const& iRef);

    RefToBaseVector& operator=(RefToBaseVector const& iRHS);
    void swap(RefToBaseVector& other);

    ~RefToBaseVector();

    //void reserve(size_type n);
    void clear();

    value_type at(size_type idx) const;
    value_type operator[](size_type idx) const;
    bool isValid() const { return holder_ != 0; }
    bool isInvalid() const { return holder_ == 0; }
    bool empty() const;
    size_type size() const;
    //size_type capacity() const;
    ProductID id() const;
    const_iterator begin() const;
    const_iterator end() const;

  private:
    holder_type* holder_;
  };
  
  template <class T>
  inline
  void
  swap(RefToBaseVector<T>& a, RefToBaseVector<T>& b) {
    a.swap(b);
  }

  template <class T>
  inline
  bool
  operator== (RefToBaseVector<T> const& a,
	      RefToBaseVector<T> const& b)
  {
    if ( a.isInvalid() && b.isInvalid() ) return true;
    if ( a.isInvalid() || b.isInvalid() ) return false;
    return  a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  //--------------------------------------------------------------------
  // Implementation of RefToBaseVector<T>
  //--------------------------------------------------------------------
  
  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector() : 
    holder_(0) 
  { }

  template <class T>
  template <class TRefVector>
  inline
  RefToBaseVector<T>::RefToBaseVector(const TRefVector& iRef) :
    holder_(new reftobase::VectorHolder<T,TRefVector>(iRef)) 
  { }

  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector(const RefToBaseVector<T>& iOther) : 
    holder_(iOther.holder_ ? iOther.holder_->clone() : 0)
  { }

  template <class T>
  inline
  RefToBaseVector<T>& 
  RefToBaseVector<T>::operator=(const RefToBaseVector& iRHS) {
    RefToBaseVector temp(iRHS);
    this->swap(temp);
    return *this;
  }

  template <class T>
  inline
  void
  RefToBaseVector<T>::swap(RefToBaseVector& other) {
    std::swap(holder_, other.holder_);
  }

  template <class T>
  inline
  RefToBaseVector<T>::~RefToBaseVector() 
  {
    delete holder_; 
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::value_type
  RefToBaseVector<T>::at(size_type idx) const 
  {
    if ( holder_ == 0 )
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "Trying to dereference null RefToBaseVector<T> in method: at(" << idx  <<")";
    return holder_->at( idx );
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::value_type
  RefToBaseVector<T>::operator[](size_type idx) const 
  {
    return at( idx ); 
  }

  template <class T>
  inline
  bool 
  RefToBaseVector<T>::empty() const 
  {
    return holder_ ? holder_->empty() : true;
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::size_type
  RefToBaseVector<T>::size() const 
  {
    return holder_ ? holder_->size() : 0;
  }

//   template <class T>
//   inline
//   typename RefToBaseVector<T>::size_type
//   RefToBaseVector<T>::capacity() const 
//   {
//     return holder_ ? holder_->capacity() : 0;
//   }


//   template <class T>
//   inline
//   void 
//   RefToBaseVector<T>::reserve(size_type n)
//   {
//     if (!holder_) holder_ = new holder_type();
//     holder_->reserve(n);
//   }

  template <class T>
  inline
  void 
  RefToBaseVector<T>::clear()
  {
    if ( holder_ != 0 )
      holder_->clear();
  }

  template <class T>
  inline
  ProductID
  RefToBaseVector<T>::id() const
  {
    if ( holder_ == 0 )
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "Trying to dereference null RefToBaseVector<T> in method: id()";
    return holder_->id();
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::const_iterator
  RefToBaseVector<T>::begin() const
  {
    if ( holder_ == 0 ) return const_iterator();
    return holder_->begin();
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::const_iterator
  RefToBaseVector<T>::end() const
  {
    if ( holder_ == 0 ) return const_iterator();
    return holder_->end();
  }

}

#endif
