#ifndef INCLUDE_ORA_PVECTOR_H
#define INCLUDE_ORA_PVECTOR_H

//
#include <vector>

namespace ora {


  /**
   * Container class replicating std::vector functionality, designed for update/append operations in CMS-ORA with no need of delete queries.
   * Write and Read are like standards vectors. Update assumes the following:
   * - updates are only applied on instances previously read from the database
   * - the changes are only performed with push_back (appending new elements) or pop_back (removing elements)
   * Only the differences from the persistent instance are updated on the database.
   */
  template <typename Tp> class PVector 
  {

    public:

    // typedefs forwarded to std::vector
    typedef typename std::vector<Tp>::size_type size_type;

    typedef typename std::vector<Tp>::const_reference const_reference;

    typedef typename std::vector<Tp>::reference reference;

    typedef typename std::vector<Tp>::const_iterator const_iterator;

    typedef typename std::vector<Tp>::iterator iterator;

    typedef typename std::vector<Tp>::const_reverse_iterator const_reverse_iterator;

    typedef typename std::vector<Tp>::reverse_iterator reverse_iterator;

    typedef typename std::vector<Tp>::value_type value_type;

    // pool specific typedef
    typedef typename std::vector<Tp> store_type;

    public:

    // default constructor
    PVector();

    // constructor
    explicit PVector(size_type n, const Tp& value=Tp());

    // copy constructor
    PVector(const PVector<Tp>&);

    // destructor
    virtual ~PVector(){
    }

    // assignment operator
    PVector<Tp>& operator=(const PVector<Tp>&);

    public:
    // methods forwarded to std::vector
    iterator begin()
    {
      return m_vec.begin();
    }

    iterator end()
    {
      return m_vec.end();
    }

    const_iterator begin() const 
    {
      return m_vec.begin();
    }

    const_iterator end() const 
    {
      return m_vec.end();
    }

    reverse_iterator rbegin()
    {
      return m_vec.rbegin();
    }

    reverse_iterator rend()
    {
      return m_vec.rend();
    }

    const_reverse_iterator rbegin() const 
    {
      return m_vec.rbegin();
    }

    const_reverse_iterator rend() const 
    {
      return m_vec.rend();
    }

    size_type size() const 
    {
      return m_vec.size();
    }

    size_type max_size() const 
    {
      return m_vec.max_size();
    }

    void resize(size_type n, const Tp& value=Tp())
    {
      m_vec.resize(n,value);
    }

    size_type capacity() const 
    {
      return m_vec.capacity();
    }

    bool empty() const 
    {
      return m_vec.empty();
    }

    void reserve(size_type n) 
    {
      m_vec.reserve(n);
    }

    reference operator[] ( size_type n ) 
    {
      return m_vec[n];
    }

    const_reference operator[] ( size_type n ) const 
    {
      return m_vec[n];
    }

    const_reference at( size_type n ) const 
    {
      return m_vec.at(n);
    }

    reference at( size_type n ) 
    {
      return m_vec.at(n);
    }

    reference front ( ) 
    {
      return m_vec.front();
    }

    const_reference front ( ) const 
    {
      return m_vec.front();
    }

    reference back ( ) 
    {
      return m_vec.back();
    }

    const_reference back ( ) const 
    {
      return m_vec.back();
    }

    void assign ( size_type n, const Tp& u ) 
    {
      m_vec.assign(n,u);
    }

    void push_back ( const Tp& x ) 
    {
      m_vec.push_back(x);
    }

    void pop_back () 
    {
      m_vec.pop_back();
    }

    void clear ( )
    {
      m_vec.clear();
    }

    // equals operator
    bool operator==(const PVector& vec) const 
    {
      return m_vec==vec.m_vec;
    }

    bool operator!=(const PVector& vec) const 
    {
      return m_vec!=vec.m_vec;
    }

    // ORA specific methods
    public:

    const void* storageAddress() const 
    {
      return &m_vec;
    }

    // access to persistent size
    size_type persistentSize() const 
    {
      return m_persistentSize;
    }

    // ORA specific attributes
    private:

    // private std::vector instance
    std::vector<Tp> m_vec;

    // persistent size
    size_type m_persistentSize;
    
  };
}


template <class Tp> ora::PVector<Tp>::PVector():m_vec(),m_persistentSize(0){
}

template <class Tp> ora::PVector<Tp>::PVector(size_type n, const Tp& value):m_vec(n,value),m_persistentSize(0){
}

template <class Tp> ora::PVector<Tp>::PVector(const PVector<Tp>& v):m_vec(v.m_vec),m_persistentSize(v.m_persistentSize){
}

template <class Tp> ora::PVector<Tp>& ora::PVector<Tp>::operator=(const PVector<Tp>& v){

  m_vec = v.m_vec;
  m_persistentSize = v.m_persistentSize;
  return *this;
}

#endif  
  
