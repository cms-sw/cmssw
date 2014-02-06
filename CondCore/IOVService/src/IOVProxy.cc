#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include <iostream>
void cond::IOVProxyData::refresh(){
  if( token.empty() ) return;
  data = dbSession.getTypedObject<cond::IOVSequence>( token );
  // loading the lazy-loading Queryable vector...
  data->loadAll();
  //**** temporary for the schema transition
  if( dbSession.isOldSchema() ){
    PoolTokenParser parser(  dbSession.storage() ); 
    data->swapTokens( parser );
  }
}

void cond::IOVProxyData::refresh( cond::DbSession& newSession ){
  dbSession = newSession;
  refresh();
}

std::pair<int,int> cond::IOVProxyData::range( cond::Time_t since, 
					      cond::Time_t  till ){
  int low = -1;
  int high = -1;
  if( data->iovs().size() ){
    cond::Time_t firstSince = data->iovs().front().sinceTime();
    if( till >= firstSince ){
      low = (since < firstSince) ? 0 :
	data->find(since)-data->iovs().begin();
      high = data->find(till) - data->iovs().begin();
      high = std::min( high+1,(int)data->iovs().size())-1;
    }
  }
  return std::make_pair( low, high );
}

void cond::IOVElementProxy::set(IOVSequence const & v, int ii) {
  if (ii>=(int)v.iovs().size() || ii<0) {
    set(cond::invalidTime, cond::invalidTime,"");
    return;
  }
  size_t i =ii;
  m_token = v.iovs()[i].token();
  m_since =  v.iovs()[i].sinceTime();
  if(i+1==v.iovs().size()) {
    m_till = v.lastTill();
    return;
  }
  cond::UnpackedTime unpackedTime;
  cond::Time_t totalNanoseconds;
  cond::Time_t totalSecondsInNanoseconds;
  switch (v.timeType()) {
  case timestamp:
    //unpacking
    unpackedTime = cond::time::unpack(v.iovs()[i+1].sinceTime());
    //number of seconds in nanoseconds (avoid multiply and divide by 1e09)
    totalSecondsInNanoseconds = ((cond::Time_t)unpackedTime.first)*1000000000;
    //total number of nanoseconds
    totalNanoseconds = totalSecondsInNanoseconds + ((cond::Time_t)(unpackedTime.second));
    //now decrementing of 1 nanosecond
    totalNanoseconds--;
    //now repacking (just change the value of the previous pair)
    unpackedTime.first = (unsigned int) (totalNanoseconds/1000000000);
    unpackedTime.second = (unsigned int)(totalNanoseconds - (cond::Time_t)unpackedTime.first*1000000000);
    m_till = cond::time::pack(unpackedTime);
    break;
  default:
    m_till = v.iovs()[i+1].sinceTime()-1;
  }
}

cond::IterHelp::IterHelp(IOVProxyData& impl) :
  lowBound(0),
  iov(&(*impl.data)), 
  elem(){
}
cond::IterHelp::IterHelp(IOVProxyData& impl, cond::Time_t lb) :
  lowBound(lb),
  iov(&(*impl.data)), 
  elem(){
}

cond::IOVRange::IOVRange():
  m_iov(),
  m_low(-1),
  m_high(-1),
  m_lowBound(0){
}
    
cond::IOVRange::IOVRange( const boost::shared_ptr<IOVProxyData>& iov, 
			  cond::Time_t since, 
			  cond::Time_t till,
			  int selection ):
  m_iov( iov ),
  m_low( -1 ),
  m_high( -1 ),
  m_lowBound(0){
  std::pair<int,int> rg = m_iov->range( since, till );
  m_low = rg.first;
  m_high = rg.second;
  if( selection > 0 ){
    m_low = rg.first;
    m_high = std::min( rg.first-selection-1, rg.second );
  } else if( selection < 0 ){
    m_low = std::max( rg.second-selection+1,rg.first);
    m_high = rg.second;
  }
  cond::Time_t tlow = m_iov->data->iovs()[m_low].sinceTime();
  if( since < tlow ) since = tlow;
  m_lowBound = since;
}

cond::IOVRange::IOVRange( const boost::shared_ptr<IOVProxyData>& iov, 
			  int selection ):
  m_iov(iov),
  m_low(-1),
  m_high(-1),
  m_lowBound(0){
  int sz  =  iov->data->iovs().size();
  m_low = 0;
  m_high = sz-1;
  if( selection > 0 ){
    m_high = std::min( selection, sz-1 );
  } else if( selection < 0 ){
    m_low = sz+selection;
    if(m_low < 0) m_low = 0;
  }
}

cond::IOVRange::IOVRange( const IOVRange& rhs ):
  m_iov( rhs.m_iov ),
  m_low( rhs.m_low ),
  m_high( rhs.m_high ),
  m_lowBound( rhs.m_lowBound){
}
    
cond::IOVRange& cond::IOVRange::operator=( const IOVRange& rhs ){
  m_iov = rhs.m_iov;
  m_low = rhs.m_low;
  m_high = rhs.m_high;
  m_lowBound = rhs.m_lowBound;
  return *this;
}

cond::IOVRange::const_iterator cond::IOVRange::find(cond::Time_t time) const {
  int n = m_iov->data->find(time)-m_iov->data->iovs().begin();
  return (n<m_low || m_high<n ) ? 
    end() :  
    boost::make_transform_iterator(boost::counting_iterator<int>(n),
				   IterHelp(*m_iov, m_lowBound));
}

cond::IOVElementProxy cond::IOVRange::front() const {
  IterHelp helper(*m_iov, m_lowBound);
  return helper( m_low );
}

cond::IOVElementProxy cond::IOVRange::back() const {
  IterHelp helper(*m_iov, m_lowBound);
  return helper( m_high );
}

size_t cond::IOVRange::size() const {
  size_t sz = 0;
  if( m_high>=0 && m_low>=0 ) sz = m_high-m_low+1;
  return sz;
}

cond::IOVProxy::IOVProxy() : 
  m_iov(){
}
 
cond::IOVProxy::~IOVProxy() {
}

cond::IOVProxy::IOVProxy(cond::DbSession& dbSession):
  m_iov(new IOVProxyData(dbSession)){
}

cond::IOVProxy::IOVProxy(cond::DbSession& dbSession,
			 const std::string & token):
  m_iov(new IOVProxyData(dbSession,token)){
}

cond::IOVProxy::IOVProxy( const boost::shared_ptr<IOVProxyData>& data ):
  m_iov( data ){
}

cond::IOVProxy::IOVProxy( const IOVProxy& rhs ):
  m_iov( rhs.m_iov ){
}

cond::IOVProxy& cond::IOVProxy::operator=( const IOVProxy& rhs ){
  m_iov = rhs.m_iov;
  return *this;
}

void cond::IOVProxy::load( const std::string & token){
  m_iov->token = token;
  m_iov->refresh();
}

bool cond::IOVProxy::refresh(){
  int oldsize = size();
  m_iov->refresh();
  return oldsize<size();
}

bool cond::IOVProxy::refresh( cond::DbSession& newSession ){
  int oldsize = size();
  m_iov->refresh( newSession );
  return oldsize<size();
}

const std::string& cond::IOVProxy::token(){
  return m_iov->token;
}

bool cond::IOVProxy::isValid( cond::Time_t currenttime ){
  return (  currenttime >= iov().firstSince() && 
	    currenttime <= iov().lastTill() );
}

std::pair<cond::Time_t, cond::Time_t> cond::IOVProxy::validity( cond::Time_t currenttime ){  
  cond::Time_t since = iov().lastTill();
  cond::Time_t till = iov().lastTill();
  IOVSequence::const_iterator iter=iov().find(currenttime);
  if (iter!=iov().iovs().end())  {
    since=iter->sinceTime();
    iter++;
    if (iter!=iov().iovs().end()) 
      till = iter->sinceTime()-1;
  }
  return std::pair<cond::Time_t, cond::Time_t>(since,till);
}

cond::IOVRange cond::IOVProxy::range(cond::Time_t since, 
				     cond::Time_t  till) const {
  return IOVRange( m_iov, since, till );
}

cond::IOVRange cond::IOVProxy::rangeHead(cond::Time_t since, 
					 cond::Time_t  till, 
					 int n) const {
  return IOVRange( m_iov, since, till, n );
}

cond::IOVRange cond::IOVProxy::rangeTail(cond::Time_t since, 
					 cond::Time_t  till, 
					 int n) const {
  return IOVRange( m_iov, since, till, -n );
}

cond::IOVRange cond::IOVProxy::head(int n) const {
  return IOVRange( m_iov, n );
}

cond::IOVRange cond::IOVProxy::tail(int n) const {
  return IOVRange( m_iov, -n );
}

cond::IOVProxy::const_iterator cond::IOVProxy::find(cond::Time_t time) const {
  int n = iov().find(time)-iov().iovs().begin();
  return ( n<0 || n>size() ) ? 
    end() :  
    boost::make_transform_iterator(boost::counting_iterator<int>(n),
				   IterHelp(*m_iov));
}

int cond::IOVProxy::size() const {
  return iov().iovs().size();
}

cond::IOVSequence const & cond::IOVProxy::iov() const {
  if( !m_iov->data.get() ) throwException( "No data found.", "IOVProxy::iov" );
  return *(m_iov->data);
}

cond::TimeType cond::IOVProxy::timetype() const {
  return iov().timeType();     
}

cond::Time_t cond::IOVProxy::firstSince() const {
  return iov().firstSince();
}

cond::Time_t cond::IOVProxy::lastTill() const {
  return iov().lastTill();
}
 
std::set<std::string> const& 
cond::IOVProxy::payloadClasses() const{
  return iov().payloadClasses();
}
  
std::string 
cond::IOVProxy::comment() const{
  return iov().comment();
}

int 
cond::IOVProxy::revision() const{
  return iov().revision();
}

cond::Time_t cond::IOVProxy::timestamp() const {
  return iov().timestamp();
}

cond::DbSession& cond::IOVProxy::db() const {
  return m_iov->dbSession;
}



