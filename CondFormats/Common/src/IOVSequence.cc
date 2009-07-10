#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/Time.h"
#include <algorithm>
#include <boost/bind.hpp>


namespace cond {
  
  
  IOVSequence::IOVSequence() : m_notOrdered(false), m_sorted(0) {}
  
  IOVSequence::IOVSequence(cond::TimeType ttype) :
    m_timetype(ttype), m_lastTill(timeTypeSpecs[ttype].endValue),
    m_notOrdered(false), m_metadata(" "),  m_sorted(0) {}
    


  IOVSequence::IOVSequence(int type, cond::Time_t till, 
			   std::string const& imetadata) :
    m_timetype(type), m_lastTill(till),m_notOrdered(false),
    m_metadata(imetadata),  m_sorted(0) {}
    
  IOVSequence::~IOVSequence(){
    delete m_sorted;
  }
  
  IOVSequence::IOVSequence(IOVSequence const & rh) : 
    m_iovs(rh.m_iovs),  
    m_timetype(rh.m_timetype),
    m_lastTill(rh.m_lastTill),
    m_notOrdered(rh.m_notOrdered),
    m_metadata(rh.m_metadata),
    m_sorted(0) {}
  
  IOVSequence & IOVSequence::operator=(IOVSequence const & rh) {
    delete m_sorted;  m_sorted=0;

    m_iovs = rh.m_iovs;  
    m_timetype = rh.m_timetype;
    m_lastTill=rh.m_lastTill;
    m_notOrdered=rh.m_notOrdered;
    m_metadata = rh.m_metadata;
    return *this;
  }


  void IOVSequence::loadAll() const {
    // m_provenance.get();
    // m_description.get();
    // m_userMetadata.get();
  }
  
  IOVSequence::Container const & IOVSequence::iovs() const {
    if (m_sorted) return *m_sorted;
    if (m_notOrdered) return sortMe();
    return m_iovs;
  }

  IOVSequence::Container const & IOVSequence::sortMe() const {
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
			  std::string const & wrapperToken) {
    if (!piovs().empty() && ( m_notOrdered || time<piovs().back().sinceTime())) disorder();
    piovs().push_back(Item(time, wrapperToken));
    return piovs().size()-1;
  }
  
  size_t IOVSequence::truncate() {
    if (m_notOrdered) disorder();
    piovs().pop_back();
    return piovs().size()-1;
  }
  


  IOVSequence::const_iterator IOVSequence::find(cond::Time_t time) const {
    if (time>=lastTill()) return iovs().end();
    IOVSequence::const_iterator p = std::upper_bound(iovs().begin(),iovs().end(),Item(time),
			    boost::bind(std::less<cond::Time_t>(),
					boost::bind(&Item::sinceTime,_1),
					boost::bind(&Item::sinceTime,_2)
					)
			    );
    return (p!=iovs().begin()) ? p-1 : iovs().end(); 
  }
  
  bool IOVSequence::exist(cond::Time_t time) const {
    IOVSequence::const_iterator p = find(time);
    return p!=iovs().end() && (*p).sinceTime()==time;

  }


  void  IOVSequence::disorder() {
    m_notOrdered=true;
    delete m_sorted; m_sorted=0;
  }

}
