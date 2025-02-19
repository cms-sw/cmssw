#ifndef INCLUDE_ORA_QUERYABLEVECTORIMPL_H
#define INCLUDE_ORA_QUERYABLEVECTORIMPL_H

#include "CondCore/ORA/interface/Exception.h"

template <class Tp> ora::RangeIterator<Tp>::RangeIterator( RangeIterator<Tp>::embedded_iterator vectorIterator):m_vecIterator(vectorIterator){
}
      
template <class Tp> ora::RangeIterator<Tp>::RangeIterator( const ora::RangeIterator<Tp>& rhs ):m_vecIterator(rhs.m_vecIterator){
}

template <class Tp> ora::RangeIterator<Tp>& ora::RangeIterator<Tp>::operator=( const ora::RangeIterator<Tp>& rhs ){
   m_vecIterator = rhs.m_vecIterator;
}

template <class Tp> ora::RangeIterator<Tp>::~RangeIterator(){
}

template <class Tp> bool ora::RangeIterator<Tp>::operator==( const ora::RangeIterator<Tp>& rhs ) const{
  return m_vecIterator == rhs.m_vecIterator;
}

template <class Tp> bool ora::RangeIterator<Tp>::operator!=( const ora::RangeIterator<Tp>& rhs ) const {
  return m_vecIterator != rhs.m_vecIterator;
}

template <class Tp> ora::RangeIterator<Tp>& ora::RangeIterator<Tp>::operator++(){
  ++m_vecIterator;
  return *this;
}

template <class Tp> ora::RangeIterator<Tp> ora::RangeIterator<Tp>::operator++(int){
  this->operator++();
  return *this;
}

template <class Tp> ora::RangeIterator<Tp> ora::RangeIterator<Tp>::operator+(int i){
  return RangeIterator(this->operator+(i));
}

template <class Tp> ora::RangeIterator<Tp> ora::RangeIterator<Tp>::operator-(int i){
  return RangeIterator(this->operator-(i));
}

template <class Tp> size_t ora::RangeIterator<Tp>::index() const {
  return m_vecIterator->first;
}

template <class Tp> const Tp* ora::RangeIterator<Tp>::operator->() const { 
  return &m_vecIterator->second; 
}
  
template <class Tp> const Tp& ora::RangeIterator<Tp>::operator*() const { 
  return m_vecIterator->second; 
}

template <class Tp> ora::RangeReverseIterator<Tp>::RangeReverseIterator( ora::RangeReverseIterator<Tp>::embedded_iterator vectorIterator):m_vecIterator(vectorIterator){
}
      
template <class Tp> ora::RangeReverseIterator<Tp>::RangeReverseIterator( const ora::RangeReverseIterator<Tp>& rhs ):m_vecIterator(rhs.m_vecIterator){
}

template <class Tp> ora::RangeReverseIterator<Tp>& ora::RangeReverseIterator<Tp>::operator=( const ora::RangeReverseIterator<Tp>& rhs ){
  m_vecIterator = rhs.m_vecIterator;
}

template <class Tp> ora::RangeReverseIterator<Tp>::~RangeReverseIterator(){
}

template <class Tp> bool ora::RangeReverseIterator<Tp>::operator==( const ora::RangeReverseIterator<Tp>& rhs ) const{
  return m_vecIterator == rhs.m_vecIterator;
}

template <class Tp> bool ora::RangeReverseIterator<Tp>::operator!=( const ora::RangeReverseIterator<Tp>& rhs ) const {
  return m_vecIterator != rhs.m_vecIterator;
}

template <class Tp> ora::RangeReverseIterator<Tp>& ora::RangeReverseIterator<Tp>::operator++(){
  ++m_vecIterator;
  return *this;
}

template <class Tp> ora::RangeReverseIterator<Tp> ora::RangeReverseIterator<Tp>::operator++(int){
  this->operator++();
  return *this;
}

template <class Tp> ora::RangeReverseIterator<Tp> ora::RangeReverseIterator<Tp>::operator+(int i){
  return RangeReverseIterator(this->operator+(i));
}

template <class Tp> ora::RangeReverseIterator<Tp> ora::RangeReverseIterator<Tp>::operator-(int i){
  return RangeReverseIterator(this->operator-(i));
}

template <class Tp> size_t ora::RangeReverseIterator<Tp>::index() const {
  return m_vecIterator->first;
}

template <class Tp> const Tp* ora::RangeReverseIterator<Tp>::operator->() const { 
  return &m_vecIterator->second; 
}

template <class Tp> const Tp& ora::RangeReverseIterator<Tp>::operator*() const { 
  return m_vecIterator->second; 
}
       
template <class Tp> ora::Range<Tp>::Range():m_data(new store_base_type ){
}

template <class Tp> ora::Range<Tp>::Range(boost::shared_ptr<store_base_type>& data):m_data(data){
}

template <class Tp> ora::Range<Tp>::Range(const ora::Range<Tp>& rhs):m_data(rhs.m_data){
}

template <class Tp> ora::Range<Tp>::~Range(){
}
      
template <class Tp> ora::Range<Tp>& ora::Range<Tp>::operator=(const ora::Range<Tp>& rhs){
  if(&rhs != this){
    m_data = rhs.m_data;
  }
  return *this;
}

template <class Tp> typename ora::Range<Tp>::iterator ora::Range<Tp>::begin() const {
  return RangeIterator<Tp>(m_data->begin());
}

template <class Tp> typename ora::Range<Tp>::iterator ora::Range<Tp>::end() const {
  return RangeIterator<Tp>(m_data->end());
}

template <class Tp> typename ora::Range<Tp>::reverse_iterator ora::Range<Tp>::rbegin() const {
  return RangeReverseIterator<Tp>(m_data->rbegin());
}

template <class Tp> typename ora::Range<Tp>::reverse_iterator ora::Range<Tp>::rend() const {
  return RangeReverseIterator<Tp>(m_data->rend());
}

template <class Tp> size_t ora::Range<Tp>::size() const {
  return m_data->size();
}

template <class Tp> size_t ora::Range<Tp>::frontIndex() const {
  return m_data->front().first;
}

template <class Tp> size_t ora::Range<Tp>::backIndex() const {
  return m_data->back().first;
}


template <class Tp> 
ora::Query<Tp>::Query(boost::shared_ptr<ora::IVectorLoader>& loader):
  m_selection(),
  m_loader(loader){
}

template <class Tp> 
ora::Query<Tp>::Query(const Query<Tp>& rhs):
  m_selection( rhs.m_selection),
  m_loader( rhs.m_loader ){
}

template <class Tp>
ora::Query<Tp>& ora::Query<Tp>::operator=(const ora::Query<Tp>& rhs){
  m_selection = rhs.m_selection;
  m_loader = rhs.m_loader;
  return *this;
}

template <class Tp>
template <typename Prim> void ora::Query<Tp>::addSelection(const std::string& dataMemberName, ora::SelectionItemType stype, Prim selectionData){
  m_selection.addDataItem(dataMemberName, stype, selectionData);
}

template <class Tp> size_t ora::Query<Tp>::count(){
  return m_loader->getSelectionCount( m_selection );
}

template <class Tp> ora::Range<Tp> ora::Query<Tp>::execute(){
  typedef typename Range<Tp>::store_base_type range_store_base_type;
  boost::shared_ptr<range_store_base_type> newData ( new range_store_base_type );
  m_loader->loadSelection( m_selection, newData.get());
  return Range<Tp>( newData );
}

template <class Tp> ora::QueryableVector<Tp>::QueryableVector():
  m_data(new PVector<Tp>),
  m_isLocked(false),
  m_isLoaded(false),
  m_loader(){
}
    
template <class Tp> ora::QueryableVector<Tp>::QueryableVector(size_t n, const Tp& value):
  m_data(new PVector<Tp>(n,value)),
  m_isLocked(false),
  m_isLoaded(false),
  m_loader(){
}
    
template <class Tp> ora::QueryableVector<Tp>::QueryableVector(const QueryableVector<Tp>& rhs):
  m_data(rhs.m_data),
  m_isLocked(rhs.m_isLocked),
  m_isLoaded(rhs.m_isLoaded),
  m_loader( rhs.m_loader ){
}

template <class Tp> ora::QueryableVector<Tp>::~QueryableVector(){
}

template <class Tp> ora::QueryableVector<Tp>& ora::QueryableVector<Tp>::operator=(const ora::QueryableVector<Tp>& rhs){
  if(&rhs != this){
    m_data = rhs.m_data;
    m_isLocked = rhs.m_isLocked;
    m_isLoaded = rhs.m_isLoaded;
    m_loader = rhs.m_loader;
  }
  return *this;
}

template <class Tp> ora::Range<Tp> ora::QueryableVector<Tp>::select(int startIndex, int endIndex) const {
  Selection sel;
  sel.addIndexItem( startIndex, endIndex );
  return select( sel );
}
    
template <class Tp> ora::Range<Tp> ora::QueryableVector<Tp>::select(const ora::Selection& sel) const {
  if(m_isLocked ){
    throwException("The Vector is locked in writing mode, cannot make queries.","ora::QueryableVector<Tp>::select");
  }
  if(!m_loader.get()){
    throwException("The Loader is not installed.","ora::QueryableVector<Tp>::select");
  }
  typedef typename Range<Tp>::store_base_type range_store_base_type;
  boost::shared_ptr<range_store_base_type> newData ( new range_store_base_type );
  m_loader->loadSelection( sel, newData.get());
  return Range<Tp>(newData);
}

template <class Tp> ora::Query<Tp> ora::QueryableVector<Tp>::query() const{
  if(m_isLocked ){
    throwException("The Vector is locked in writing mode, cannot make queries.","ora::QueryableVector<Tp>::query");
  }
  if(!m_loader.get()){
    throwException("The Loader is not installed.","ora::QueryableVector<Tp>::query");
  }
  return Query<Tp>(m_loader);
}
    
template <class Tp> bool ora::QueryableVector<Tp>::lock() {
  bool wasLocked = m_isLocked;
  m_isLocked = true;
  return wasLocked;
}
        
template <class Tp> bool ora::QueryableVector<Tp>::isLocked() const {
  return m_isLocked;
}

template <class Tp> typename ora::QueryableVector<Tp>::iterator ora::QueryableVector<Tp>::begin(){
  initialize();
  return m_data->begin();
}

template <class Tp> typename ora::QueryableVector<Tp>::iterator ora::QueryableVector<Tp>::end(){
  initialize();
  return m_data->end();
}

template <class Tp> typename ora::QueryableVector<Tp>::const_iterator ora::QueryableVector<Tp>::begin() const {
  initialize();
  return m_data->begin();
}

template <class Tp> typename ora::QueryableVector<Tp>::const_iterator ora::QueryableVector<Tp>::end() const {
  initialize();
  return m_data->end();
}

template <class Tp> typename ora::QueryableVector<Tp>::reverse_iterator ora::QueryableVector<Tp>::rbegin(){
  initialize();
  return m_data->rbegin();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::reverse_iterator ora::QueryableVector<Tp>::rend(){
  initialize();
  return m_data->rend();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reverse_iterator ora::QueryableVector<Tp>::rbegin() const {
  initialize();
  return m_data->rbegin();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reverse_iterator ora::QueryableVector<Tp>::rend() const {
  initialize();
  return m_data->rend();
}

template <class Tp> size_t ora::QueryableVector<Tp>::size() const {
  initialize();
  return m_data->size();
}

template <class Tp> size_t ora::QueryableVector<Tp>::max_size() const {
  initialize();
  return m_data->max_size();
}
    
template <class Tp> void ora::QueryableVector<Tp>::resize(size_t n, const Tp& value){
  initialize();
  m_data->resize(n,value);
}
    
template <class Tp> size_t ora::QueryableVector<Tp>::capacity() const {
  initialize();
  return m_data->capacity();
}
    
template <class Tp> bool ora::QueryableVector<Tp>::empty() const {
  initialize();
  return m_data->empty();
}
    
template <class Tp> void ora::QueryableVector<Tp>::reserve(size_t n) {
  initialize();
  m_data->reserve(n);
}

template <class Tp> typename ora::QueryableVector<Tp>::reference ora::QueryableVector<Tp>::operator[] ( size_t n ){
  initialize();
  return m_data->operator[](n);
}

template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::operator[] ( size_t n ) const {
  initialize();
  return m_data->operator[](n);
}

template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::at( size_t n ) const {
  initialize();
  return m_data->operator[](n);
}
    
template <class Tp> typename ora::QueryableVector<Tp>::reference ora::QueryableVector<Tp>::at( size_t n ) {
  initialize();
  return m_data->operator[](n);
}

template <class Tp> typename ora::QueryableVector<Tp>::reference ora::QueryableVector<Tp>::front ( ) {
  initialize();
  return m_data->front();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::front ( ) const {
  initialize();
  return m_data->front();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::reference ora::QueryableVector<Tp>::back ( ) {
  return m_data->back();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::back ( ) const {
  initialize();
  return m_data->back();
}

template <class Tp> void ora::QueryableVector<Tp>::assign ( size_t n, const Tp& u ) {
  initialize();
  m_data->assign(n,u);
}

template <class Tp> void ora::QueryableVector<Tp>::push_back ( const Tp& x ){
  initialize();
  m_isLocked = true;
  m_data->push_back(x);
}

template <class Tp> void ora::QueryableVector<Tp>::pop_back (){
  initialize();
  m_isLocked = true;
  m_data->pop_back();
}

template <class Tp> void ora::QueryableVector<Tp>::clear ( ){
  initialize();
  m_data->clear();
  m_isLoaded = false;
}

template <class Tp> void ora::QueryableVector<Tp>::reset ( ){
  initialize();
  m_data->clear();
  m_isLoaded = false;
  m_isLocked = false;
}

template <class Tp> bool ora::QueryableVector<Tp>::operator==(const ora::QueryableVector<Tp>& vec) const {
  initialize();
  vec.initialize();
  return m_data->operator==(*vec.m_data);
}

template <class Tp> bool ora::QueryableVector<Tp>::operator!=(const ora::QueryableVector<Tp>& vec) const {
  initialize();
  vec.initialize();
  return m_data->operator!=(*vec.m_data);
}

template <class Tp> size_t ora::QueryableVector<Tp>::persistentSize() const {
  // not sure needs init...
  //initialize();
  return m_data->persistentSize();
}

template <class Tp> const void* ora::QueryableVector<Tp>::storageAddress() const {
  return m_data.get();
}

template <class Tp> void ora::QueryableVector<Tp>::load() const {
  initialize();
}

template <class Tp> void ora::QueryableVector<Tp>::initialize() const {
  if(m_loader.get() && !m_isLocked && !m_isLoaded){
    m_loader->load(m_data.get());
    m_isLoaded = true;
  }
}

#endif  // 
