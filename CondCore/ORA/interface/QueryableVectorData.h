#ifndef INCLUDE_ORA_QUERYABLEVECTORDATA_H
#define INCLUDE_ORA_QUERYABLEVECTORDATA_H

#include "PVector.h"
//
#include <boost/shared_ptr.hpp>
//
    #include <iostream>

namespace ora    {

  class Selection;
  
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

    // notify the underlying storage system that the embedded object has been destructed.
    // maybe not required...
    //virtual void notify() const=0;

    // invalidates the current loader. Called by the underlying service at his destruction time.
    virtual void invalidate()=0;

    // queries the validity of the current relation with the underlying storage system
    virtual bool isValid() const=0;

  };

  
  
  template <typename Tp> class Iterator {
    
    public:
    Iterator( typename ora::PVector<std::pair<size_t,Tp> >::iterator vectorIterator):m_vecIterator(vectorIterator){
    }
      
    Iterator( const Iterator& rhs ):m_vecIterator(rhs.m_vecIterator){
    }

    Iterator& operator=( const Iterator& rhs ){
      m_vecIterator = rhs.m_vecIterator;
    }
    virtual ~Iterator(){
    }

    bool operator==( const Iterator& rhs ) const{
      return m_vecIterator == rhs.m_vecIterator;
    }

    bool operator!=( const Iterator& rhs ) const {
      return m_vecIterator != rhs.m_vecIterator;
    }

    Iterator& operator++(){
      ++m_vecIterator;
      return *this;
    }

    Iterator operator++(int){
      this->operator++();
      return *this;
    }

    size_t index(){
      return m_vecIterator->first;
    }

    Tp* operator->() { return &m_vecIterator->second; }
    Tp& operator*() { return m_vecIterator->second; }

    private:
    typename ora::PVector<std::pair<size_t, Tp> >::iterator m_vecIterator;

  };

  template <typename Tp> class CIterator {

    public:
    CIterator( typename ora::PVector<std::pair<size_t,Tp> >::const_iterator vectorIterator):m_vecIterator(vectorIterator){
    }
      
    CIterator( const CIterator& rhs ):m_vecIterator(rhs.m_vecIterator){
    }

    CIterator& operator=( const CIterator& rhs ){
      m_vecIterator = rhs.m_vecIterator;
    }

    virtual ~CIterator(){
    }

    bool operator==( const CIterator& rhs ) const{
      return m_vecIterator == rhs.m_vecIterator;
    }

    bool operator!=( const CIterator& rhs ) const {
      return m_vecIterator != rhs.m_vecIterator;
    }

    CIterator& operator++(){
      ++m_vecIterator;
      return *this;
    }

    CIterator operator++(int){
      this->operator++();
      return *this;
    }

    size_t index(){
      return m_vecIterator->first;
    }

    const Tp* operator->() const { return &m_vecIterator->second; }
    const Tp& operator*() const { return m_vecIterator->second; }

    private:
    typename ora::PVector<std::pair<size_t, Tp> >::const_iterator m_vecIterator;
  };

  template <typename Tp> class RIterator {

    public:
    RIterator( typename ora::PVector<std::pair<size_t,Tp> >::reverse_iterator vectorIterator):m_vecIterator(vectorIterator){
    }
      
    RIterator( const RIterator& rhs ):m_vecIterator(rhs.m_vecIterator){
    }

    RIterator& operator=( const RIterator& rhs ){
      m_vecIterator = rhs.m_vecIterator;
    }

    virtual ~RIterator(){
    }

    bool operator==( const RIterator& rhs ) const{
      return m_vecIterator == rhs.m_vecIterator;
    }

    bool operator!=( const RIterator& rhs ) const {
      return m_vecIterator != rhs.m_vecIterator;
    }

    RIterator& operator++(){
      ++m_vecIterator;
      return *this;
    }

    RIterator operator++(int){
      this->operator++();
      return *this;
    }

    size_t index(){
      return m_vecIterator->first;
    }

    Tp* operator->() { return &m_vecIterator->second; }
    Tp& operator*() { return m_vecIterator->second; }

    private:
    typename ora::PVector<std::pair<size_t, Tp> >::reverse_iterator m_vecIterator;
  };

  template <typename Tp> class CRIterator {

    public:
    CRIterator( typename ora::PVector<std::pair<size_t,Tp> >::const_reverse_iterator vectorIterator):m_vecIterator(vectorIterator){
    }
      
    CRIterator( const CRIterator& rhs ):m_vecIterator(rhs.m_vecIterator){
    }

    CRIterator& operator=( const CRIterator& rhs ){
      m_vecIterator = rhs.m_vecIterator;
    }

    virtual ~CRIterator(){
    }

    bool operator==( const CRIterator& rhs ) const{
      return m_vecIterator == rhs.m_vecIterator;
    }

    bool operator!=( const CRIterator& rhs ) const {
      return m_vecIterator != rhs.m_vecIterator;
    }

    CRIterator& operator++(){
      ++m_vecIterator;
      return *this;
    }

    CRIterator operator++(int){
      this->operator++();
      return *this;
    }

    size_t index(){
      return m_vecIterator->first;
    }

    const Tp* operator->() const { return &m_vecIterator->second; }
    const Tp& operator*() const { return m_vecIterator->second; }
       
    private:
    typename ora::PVector<std::pair<size_t, Tp> >::const_reverse_iterator m_vecIterator;
  };

  class IVectorData {
    public:
    virtual ~IVectorData(){
    }

    virtual const void* storageAddress() const=0;
  };

  template <typename Tp> class QueryableVectorData: public IVectorData {
    
    public:

    typedef Iterator<Tp> iterator;
    typedef CIterator<Tp> const_iterator;
    typedef RIterator<Tp> reverse_iterator;
    typedef CRIterator<Tp> const_reverse_iterator;
    typedef typename std::pair<size_t,Tp> store_item_type;
    typedef ora::PVector<std::pair<size_t,Tp> > store_base_type;
    typedef typename ora::PVector<std::pair<size_t,Tp> >::store_type store_type;

    public:

    QueryableVectorData():IVectorData(),m_vec(){}
    QueryableVectorData(size_t n, const Tp& value=Tp()):IVectorData(),m_vec(n,std::make_pair(0,value)){}
    
    iterator begin(){
      return iterator(m_vec.begin());
    }
        
    iterator end(){
      return iterator(m_vec.end());
    }
    
    const_iterator cbegin() const {
      return const_iterator(m_vec.begin());
    }
        
    const_iterator cend() const {
      return const_iterator(m_vec.end());
    }

    reverse_iterator rbegin(){
      return reverse_iterator(m_vec.rbegin());
    }
        
    reverse_iterator rend(){
      return reverse_iterator(m_vec.rend());
    }
    
    const_reverse_iterator crbegin() const {
      return const_reverse_iterator(m_vec.rbegin());
    }
        
    const_reverse_iterator crend() const {
      return const_reverse_iterator(m_vec.rend());
    }

    const Tp& operator[](size_t n) const { return m_vec[n].second; }
    Tp& operator[](size_t n) { return m_vec[n].second; }

    const Tp& back() const { return m_vec.back().second; }
    Tp& back() { return m_vec.back().second; }
    
    const Tp& front() const { return m_vec.front().second; }
    Tp& front() { return m_vec.front().second; }

    void assign ( size_t n, const Tp& u ) {
      for(size_t i=0;i<n;i++) push_back(u);
    }

    void push_back ( const Tp& x ){
      m_vec.push_back(std::make_pair(m_vec.size(),x));
    }

    void pop_back (){
      m_vec.pop_back();
    }
    
    size_t size() const {
      return m_vec.size();
    }

    size_t max_size() const {
      return m_vec.max_size();
    }
    
    void resize(size_t n, const Tp& value=Tp()){
      size_t sz = size();
      for(size_t i=sz;i>n;i--) pop_back();
      for(size_t i=n;i>sz;i--) push_back(value);
    }
    
    size_t capacity() const {
      return m_vec.capacity();
    }
    
    bool empty() const {
      return m_vec.empty();
    }
    
    void reserve(size_t n) {
      m_vec.reserve(n);
    }

    void clear(){
      m_vec.clear();
    }
    
    bool operator==(const QueryableVectorData& rhs) const { return m_vec==rhs.m_vec; }
    bool operator!=(const QueryableVectorData& rhs) const { return m_vec!=rhs.m_vec; }

    const void* storageAddress() const {
      std::cout << "### calling storage addresse ="<<&m_vec<<std::endl;
      return &m_vec;
    }

    size_t persistentSize() const {
      return m_vec.persistentSize();
    }

    //...
    private:

    // private vector
    store_base_type m_vec;
  };

  class LoaderClient {

    public:
    LoaderClient();

    virtual ~LoaderClient();

    explicit LoaderClient(boost::shared_ptr<IVectorLoader>& loader);
    
    LoaderClient( const LoaderClient& rhs );

    LoaderClient& operator=(const LoaderClient& rhs );

    bool hasLoader() const;
    
    boost::shared_ptr<IVectorLoader> loader() const;

    void reset();

    void install(boost::shared_ptr<IVectorLoader>& loader);    

    private:
    mutable boost::shared_ptr<IVectorLoader> m_loader;
    
  };
  
}

#endif  // 
