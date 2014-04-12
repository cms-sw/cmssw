#ifndef INCLUDE_ORA_QUERYABLEVECTOR_H
#define INCLUDE_ORA_QUERYABLEVECTOR_H

#include "Selection.h"
#include "PVector.h"
// externals
#include <boost/shared_ptr.hpp>

namespace ora {

  class IVectorLoader {
    public:

    // destructor
    virtual ~IVectorLoader(){
    }

    public:

    // triggers the data loading
    virtual bool load(void* address) const=0;

    virtual bool loadSelection(const ora::Selection& selection, void* address) const=0;

    virtual size_t getSelectionCount( const ora::Selection& selection ) const=0;

    // invalidates the current loader. Called by the underlying service at his destruction time.
    virtual void invalidate()=0;

    // queries the validity of the current relation with the underlying storage system
    virtual bool isValid() const=0;

  };
  
  template <typename Tp> class RangeIterator {
    public:
    typedef typename std::vector<std::pair<size_t,Tp> >::const_iterator embedded_iterator;

    public:
    RangeIterator( embedded_iterator vectorIterator);
      
    RangeIterator( const RangeIterator& rhs );

    RangeIterator& operator=( const RangeIterator& rhs );

    virtual ~RangeIterator();

    bool operator==( const RangeIterator& rhs ) const;

    bool operator!=( const RangeIterator& rhs ) const;

    RangeIterator& operator++();

    RangeIterator operator++(int);

    RangeIterator operator+(int i);

    RangeIterator operator-(int i);

    size_t index() const;

    const Tp* operator->() const;
    const Tp& operator*() const;

    private:
    embedded_iterator m_vecIterator;
  };

  template <typename Tp> class RangeReverseIterator {
    public:
    typedef typename std::vector<std::pair<size_t,Tp> >::const_reverse_iterator embedded_iterator;

    public:
    RangeReverseIterator( embedded_iterator vectorIterator);
      
    RangeReverseIterator( const RangeReverseIterator& rhs );

    RangeReverseIterator& operator=( const RangeReverseIterator& rhs );

    virtual ~RangeReverseIterator();

    bool operator==( const RangeReverseIterator& rhs ) const;

    bool operator!=( const RangeReverseIterator& rhs ) const;

    RangeReverseIterator& operator++();

    RangeReverseIterator operator++(int);

    RangeReverseIterator operator+(int i);

    RangeReverseIterator operator-(int i);

    size_t index() const;

    const Tp* operator->() const;
    const Tp& operator*() const;
       
    private:
    embedded_iterator m_vecIterator;
  };

  template <typename Tp> class Range {
    public:
    typedef const Tp& reference;
    typedef RangeIterator<Tp> iterator;
    typedef RangeReverseIterator<Tp> reverse_iterator;
    typedef std::vector<std::pair<size_t,Tp> > store_base_type;

    public:
    Range();

    explicit Range(boost::shared_ptr<store_base_type>& data);

    Range(const Range& rhs);

    virtual ~Range();

    Range& operator=(const Range& rhs);

    iterator begin() const;

    iterator end() const;

    reverse_iterator rbegin() const;

    reverse_iterator rend() const;

    size_t size() const;

    size_t frontIndex() const;

    size_t backIndex() const;
      
    private:

    boost::shared_ptr<store_base_type> m_data;
  };
  
  template <typename Tp> class Query {
    public:
    explicit Query(boost::shared_ptr<IVectorLoader>& loader);

    Query(const Query<Tp>& rhs);
    
    Query& operator=(const Query<Tp>& rhs);

    virtual ~Query(){
    }
    
    template <typename Prim> void addSelection(const std::string& dataMemberName, SelectionItemType stype, Prim selectionData);

    size_t count();
    
    Range<Tp> execute();

    private:
    Selection m_selection;
    boost::shared_ptr<IVectorLoader> m_loader;
  };
  
  template <typename Tp> class QueryableVector {
 
    public:

    // typedefs forwarded to std::vector
    typedef typename PVector<Tp>::size_type size_type;
    typedef typename PVector<Tp>::const_reference const_reference;
    typedef typename PVector<Tp>::reference reference;
    typedef typename PVector<Tp>::const_iterator const_iterator;
    typedef typename PVector<Tp>::iterator iterator;
    typedef typename PVector<Tp>::const_reverse_iterator const_reverse_iterator;
    typedef typename PVector<Tp>::reverse_iterator reverse_iterator;
    typedef typename PVector<Tp>::value_type value_type;

    // ora specific typedef
    typedef PVector<Tp> store_base_type;
    //typedef typename PVector<Tp>::store_type store_type;
    typedef std::vector<std::pair<size_t,Tp> > range_store_base_type;

    public:
    // default constructor
    QueryableVector();
    
    // constructor
    explicit QueryableVector(size_t n, const Tp& value=Tp());
    
    // copy constructor: not sure what to do...
    QueryableVector(const QueryableVector<Tp>& rhs);

    // destructor
    virtual ~QueryableVector();

    // assignment operator: not sure what to do...
    QueryableVector<Tp>& operator=(const QueryableVector<Tp>& rhs);

    public:

    Range<Tp> select(int startIndex, int endIndex=Selection::endOfRange) const;
    
    Range<Tp> select(const Selection& sel) const;

    Query<Tp> query() const;
    
    bool lock();
        
    bool isLocked() const;

    public:
    
    iterator begin();

    iterator end();

    const_iterator begin() const;

    const_iterator end() const;

    reverse_iterator rbegin();
    
    reverse_iterator rend();
    
    const_reverse_iterator rbegin() const;
    
    const_reverse_iterator rend() const;

    size_t size() const;

    size_t max_size() const;
    
    void resize(size_t n, const Tp& value=Tp());
    
    size_t capacity() const;
    
    bool empty() const;
    
    void reserve(size_t n);

    reference operator[] ( size_t n );

    const_reference operator[] ( size_t n ) const;

    reference at( size_t n );

    const_reference at( size_t n ) const;
    
    reference front ( );
    
    const_reference front ( ) const;
    
    reference back ( );
    
    const_reference back ( ) const;

    void assign ( size_t n, const Tp& u );

    void push_back ( const Tp& x );

    void pop_back ();

    void clear ( );

    void reset ( );

    // equals operator
    bool operator==(const QueryableVector& vec) const;
    
    bool operator!=(const QueryableVector& vec) const;

    public:
    // access to persistent size
    size_t persistentSize() const;

    const void* storageAddress() const;

    void load() const;

    private:
    void initialize() const;

    private:
    boost::shared_ptr<store_base_type> m_data;
    bool m_isLocked;
    mutable bool m_isLoaded;
    mutable boost::shared_ptr<IVectorLoader> m_loader;

};

}

#include "QueryableVectorImpl.h"

#endif  
