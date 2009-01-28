#include "CondFormats/Common/interface/IOVSequence.h"
#include <algorithm>
#include <boost/bind.hpp>


namespace cond {
  
  
  IOVSequence::IOVSequence(){}
  
  IOVSequence::IOVSequence(int type, cond::Time_t till, 
			   std::string const& imetadata) :
    m_timetype(type), m_lastTill(till),m_notOrdered(false),
    m_metadata(imetadata){}
    
  IOVSequence::~IOVSequence(){}
  
  
  size_t IOVSequence::add(cond::Time_t time, 
			  std::string const & wrapperToken) {
    if (!iovs().empty() && ( m_notOrdered || time<iovs().back().sinceTime())) disorder();
    iovs().push_back(Item(time, wrapperToken));
    return iovs().size()-1;
  }
  
  
  IOVSequence::const_iterator IOVSequence::find(cond::Time_t time) const {
    IOVSequence::const_iterator p = std::upper_bound(iovs().begin(),iovs().end(),Item(time),
			    boost::bind(std::less<cond::Time_t>(),
					boost::bind(&Item::sinceTime,_1),
					boost::bind(&Item::sinceTime,_2)
					)
			    );
    return (p!=iovs().begin()) ? p-1 : iovs().end(); 
  }
  

  void  IOVSequence::disorder() {
    m_notOrdered=true;
    // delete m_sorted;
  }

}
