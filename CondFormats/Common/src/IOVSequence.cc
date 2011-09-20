#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/Time.h"
#include <algorithm>
#include <boost/bind.hpp>

namespace cond {
  
  IOVSequence::IOVSequence() : 
    m_iovs(),
    m_timetype(-1),
    m_lastTill(0),
    m_notOrdered(false), 
    m_metadata(""),
    m_payloadClasses(),
    m_scope( Unknown ),
    m_sorted(0) {}
  
  IOVSequence::IOVSequence( cond::TimeType ttype ) :
    m_iovs(),
    m_timetype(ttype),
    m_lastTill(timeTypeSpecs[ttype].endValue),
    m_notOrdered(false), 
    m_metadata(" "),
    m_payloadClasses(),
    m_scope( Unknown ),
    m_sorted(0) {}

  IOVSequence::IOVSequence(int ttype, 
                           cond::Time_t till, 
			   std::string const& imetadata) :
    m_iovs(),
    m_timetype(ttype),
    m_lastTill(till),
    m_notOrdered(false), 
    m_metadata(imetadata),
    m_payloadClasses(),
    m_scope( Unknown ),
    m_sorted(0) {}
    
  IOVSequence::~IOVSequence(){
    delete m_sorted;
  }
  
  IOVSequence::IOVSequence(IOVSequence const & rh) : 
    UpdateStamp(rh),
    m_iovs(rh.m_iovs),  
    m_timetype(rh.m_timetype),
    m_lastTill(rh.m_lastTill),
    m_notOrdered(rh.m_notOrdered),
    m_metadata(rh.m_metadata),
    m_payloadClasses(rh.m_payloadClasses),
    m_scope( rh.m_scope ),
    m_sorted(0) {}
  
  IOVSequence & IOVSequence::operator=(IOVSequence const & rh) {
    delete m_sorted;  m_sorted=0;

    m_iovs = rh.m_iovs;  
    m_timetype = rh.m_timetype;
    m_lastTill=rh.m_lastTill;
    m_notOrdered=rh.m_notOrdered;
    m_metadata = rh.m_metadata;
    m_payloadClasses = rh.m_payloadClasses;
    m_scope = rh.m_scope;
    return *this;
  }


  void IOVSequence::loadAll() const {
    // m_provenance.get();
    // m_description.get();
    // m_userMetadata.get();
    m_iovs.load();
  }
  
  IOVSequence::Container const & IOVSequence::iovs() const {
    if (m_sorted) return *m_sorted;
    if (m_notOrdered) return sortMe();
    return m_iovs;
  }

  IOVSequence::Container const & IOVSequence::sortMe() const {
    m_iovs.load();
    delete m_sorted; // shall not be necessary;
    Container * local = new Container(m_iovs);
    std::sort(local->begin(), local->end(), boost::bind(std::less<cond::Time_t>(),
							boost::bind(&Item::sinceTime,_1),
							boost::bind(&Item::sinceTime,_2)
							) );
    m_sorted = local;
    return *m_sorted;
  }


  size_t IOVSequence::add(cond::Time_t time, 
			  std::string const & token,
                          std::string const & payloadClassName ) {
    if (!piovs().empty() && ( m_notOrdered || time<piovs().back().sinceTime())) disorder();
    piovs().push_back(Item(time, token));
    m_payloadClasses.insert( payloadClassName );
    return piovs().size()-1;
  }
  
  size_t IOVSequence::truncate() {
    if (m_notOrdered) disorder();
    piovs().pop_back();
    return piovs().size()-1;
  }

  IOVSequence::const_iterator IOVSequence::find(cond::Time_t time) const {
    if (time>lastTill()) return iovs().end();
    IOVSequence::const_iterator p = std::upper_bound(iovs().begin(),iovs().end(),Item(time),
			    boost::bind(std::less<cond::Time_t>(),
					boost::bind(&Item::sinceTime,_1),
					boost::bind(&Item::sinceTime,_2)
					)
			    );
    return (p!=iovs().begin()) ? p-1 : iovs().end(); 
  }
  

  IOVSequence::const_iterator IOVSequence::findSince(cond::Time_t time) const {
    IOVSequence::const_iterator p = find(time);
    return (p!=iovs().end() && (*p).sinceTime()==time) ? p : iovs().end();
  }
  
  bool IOVSequence::exist(cond::Time_t time) const {
    return findSince(time)!=iovs().end();
  }

  void IOVSequence::updateMetadata( const std::string& metadata, 
				    bool append ){
    std::string sep(". ");
    if( !metadata.empty() ){
      if (append && !m_metadata.empty()) {
	m_metadata += sep + metadata;
      }
      else m_metadata = metadata;
    }
  }

  void  IOVSequence::disorder() {
    m_notOrdered=true;
    delete m_sorted; m_sorted=0;
  }

  void IOVSequence::swapTokens( ora::ITokenParser& parser ) const {
    for( IOVSequence::const_iterator iT = m_iovs.begin();
         iT != m_iovs.end(); ++iT ){
      iT->swapToken( parser );
      // adding the classname 'by hand'
      std::string className = parser.className( iT->token() );
      const_cast<IOVSequence* >(this)->m_payloadClasses.insert( className );
    }
  }

  void IOVSequence::swapOIds( ora::ITokenWriter& writer ) const {
    for( IOVSequence::const_iterator iT = m_iovs.begin();
         iT != m_iovs.end(); ++iT ){
      iT->swapOId( writer );
    }
  }

}

