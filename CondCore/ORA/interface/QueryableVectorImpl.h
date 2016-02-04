#ifndef INCLUDE_ORA_QUERYABLEVECTORIMPL_H
#define INCLUDE_ORA_QUERYABLEVECTORIMPL_H

template <class Tp> ora::Range<Tp>::Range():m_data(new QueryableVectorData<Tp>){
}

template <class Tp> ora::Range<Tp>::Range(boost::shared_ptr<ora::QueryableVectorData<Tp> >& data):m_data(data){
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

template <class Tp> typename ora::Range<Tp>::const_iterator ora::Range<Tp>::begin() const {
  return m_data->cbegin();
}

template <class Tp> typename ora::Range<Tp>::const_iterator ora::Range<Tp>::end() const {
  return m_data->cend();
}

template <class Tp> typename ora::Range<Tp>::const_reverse_iterator ora::Range<Tp>::rbegin() const {
  return m_data->crbegin();
}

template <class Tp> typename ora::Range<Tp>::const_reverse_iterator ora::Range<Tp>::rend() const {
  return m_data->crend();
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



template <class Tp> ora::Query<Tp>::Query(boost::shared_ptr<ora::IVectorLoader>& loader):LoaderClient(loader){
}

template <class Tp>
template <typename Prim> void ora::Query<Tp>::addSelection(const std::string& dataMemberName, ora::SelectionItemType stype, Prim selectionData){
  m_selection.addDataItem(dataMemberName, stype, selectionData);
}

template <class Tp> size_t ora::Query<Tp>::count(){
  return loader()->getSelectionCount( m_selection );
}

template <class Tp> ora::Range<Tp> ora::Query<Tp>::execute(){
  boost::shared_ptr<QueryableVectorData<Tp> > newData ( new QueryableVectorData<Tp> );
  loader()->loadSelection( m_selection, const_cast<void*>(newData->storageAddress()) );
  return Range<Tp>( newData );
}

template <class Tp> ora::QueryableVector<Tp>::QueryableVector():LoaderClient(),m_data(new QueryableVectorData<Tp>),m_isLocked(false),m_isLoaded(false){
}
    
template <class Tp> ora::QueryableVector<Tp>::QueryableVector(size_t n, const Tp& value):LoaderClient(),m_data(new QueryableVectorData<Tp>(n,value)),m_isLocked(false),m_isLoaded(false){
}
    
template <class Tp> ora::QueryableVector<Tp>::QueryableVector(const QueryableVector<Tp>& rhs):LoaderClient(rhs),m_data(rhs.m_data),m_isLocked(rhs.m_isLocked),m_isLoaded(rhs.m_isLoaded){
}

template <class Tp> ora::QueryableVector<Tp>::~QueryableVector(){
}

template <class Tp> ora::QueryableVector<Tp>& ora::QueryableVector<Tp>::operator=(const ora::QueryableVector<Tp>& rhs){
  if(&rhs != this){
    m_data = rhs.m_data;
    m_isLocked = rhs.m_isLocked;
    m_isLoaded = rhs.m_isLoaded;
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
  if(!hasLoader()){
    throwException("The Loader is not installed.","ora::QueryableVector<Tp>::select");
  }
  boost::shared_ptr<QueryableVectorData<Tp> > newData ( new QueryableVectorData<Tp> );
  loader()->loadSelection( sel, const_cast<void*>(newData->storageAddress()) );
  return Range<Tp>(newData);
}

template <class Tp> ora::Query<Tp> ora::QueryableVector<Tp>::query() const{
  if(m_isLocked ){
    throwException("The Vector is locked in writing mode, cannot make queries.","ora::QueryableVector<Tp>::query");
  }
  if(!hasLoader()){
    throwException("The Loader is not installed.","ora::QueryableVector<Tp>::query");
  }
  boost::shared_ptr<IVectorLoader> loaderH = loader();
  return Query<Tp>(loaderH);
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
  return m_data->cbegin();
}

template <class Tp> typename ora::QueryableVector<Tp>::const_iterator ora::QueryableVector<Tp>::end() const {
  initialize();
  return m_data->cend();
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
  return m_data->crbegin();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reverse_iterator ora::QueryableVector<Tp>::rend() const {
  return m_data->crend();
}

template <class Tp> size_t ora::QueryableVector<Tp>::size() const {
  initialize();
  return m_data->size();
}

template <class Tp> size_t ora::QueryableVector<Tp>::max_size() const {
  return m_data->max_size();
}
    
template <class Tp> void ora::QueryableVector<Tp>::resize(size_t n, const Tp& value){
  initialize();
  m_data->resize(n,value);
}
    
template <class Tp> size_t ora::QueryableVector<Tp>::capacity() const {
  return m_data->capacity();
}
    
template <class Tp> bool ora::QueryableVector<Tp>::empty() const {
  initialize();
  return m_data->empty();
}
    
template <class Tp> void ora::QueryableVector<Tp>::reserve(size_t n) {
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
  return m_data->front();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::front ( ) const {
  return m_data->front();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::reference ora::QueryableVector<Tp>::back ( ) {
  return m_data->back();
}
    
template <class Tp> typename ora::QueryableVector<Tp>::const_reference ora::QueryableVector<Tp>::back ( ) const {
  return m_data->back();
}

template <class Tp> void ora::QueryableVector<Tp>::assign ( size_t n, const Tp& u ) {
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
  m_data->clear();
  m_isLoaded = false;
}

template <class Tp> void ora::QueryableVector<Tp>::reset ( ){
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
  return m_data->storageAddress();
}

template <class Tp> void ora::QueryableVector<Tp>::load() const {
  initialize();
}

template <class Tp> void ora::QueryableVector<Tp>::initialize() const {
  if(hasLoader() && !m_isLocked && !m_isLoaded){
    loader()->load(const_cast<void*>(m_data->storageAddress()));
    m_isLoaded = true;
  }
}

#endif  // 
