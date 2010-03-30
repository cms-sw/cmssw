#ifndef INCLUDE_ORA_QUERYABLEVECTOR_H
#define INCLUDE_ORA_QUERYABLEVECTOR_H

#include "QueryableVectorData.h"
#include "Selection.h"

namespace ora {
  
  template <typename Tp> class Range {
    public:

    typedef const Tp& const_reference;
    typedef CIterator<Tp> const_iterator;
    typedef CRIterator<Tp> const_reverse_iterator;

    public:
    Range();

    explicit Range(boost::shared_ptr<QueryableVectorData<Tp> >& data);

    Range(const Range& rhs);

    virtual ~Range();

    Range& operator=(const Range& rhs);

    const_iterator begin() const;

    const_iterator end() const;

    const_reverse_iterator rbegin() const;

    const_reverse_iterator rend() const;

    size_t size() const;

    size_t frontIndex() const;

    size_t backIndex() const;
      
    private:

    boost::shared_ptr<QueryableVectorData<Tp> > m_data;
  };
  
  template <typename Tp> class Query: public LoaderClient {
    public:
    explicit Query(boost::shared_ptr<IVectorLoader>& loader);
    
    virtual ~Query(){
    }
    
    template <typename Prim> void addSelection(const std::string& dataMemberName, SelectionItemType stype, Prim selectionData);

    size_t count();
    
    Range<Tp> execute();

    private:
    Selection m_selection;
  };
  
  template <typename Tp> class QueryableVector: public LoaderClient {
 
    public:

    // std::vector like typedefs
    typedef Tp& reference;
    typedef const Tp& const_reference;
    typedef Iterator<Tp> iterator;
    typedef CIterator<Tp> const_iterator;
    typedef RIterator<Tp> reverse_iterator;
    typedef CRIterator<Tp> const_reverse_iterator;
    typedef typename std::vector<Tp>::value_type value_type;
    // pool specific typedefs    
    typedef Query<Tp> pquery;
    typedef Range<Tp> prange;
    typedef typename QueryableVectorData<Tp>::store_item_type store_item_type;
    typedef typename QueryableVectorData<Tp>::store_base_type store_base_type;
    typedef typename QueryableVectorData<Tp>::store_type store_type;

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

    boost::shared_ptr<QueryableVectorData<Tp> > m_data;
    bool m_isLocked;
    mutable bool m_isLoaded;

};

}

#include "QueryableVectorImpl.h"

#endif  
