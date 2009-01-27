#ifndef Cond_IOVSequence_h
#define Cond_IOVSequence_h
#include "CondFormats/Common/interface/UpdateStamp.h"
#include "CondFormats/Common/interface/IOVElement.h"
#include "CondFormats/Common/interface/Time.h"
#include <vector>
#include <string>
#include "POOLCore/PVector.h"


namespace cond {

  /** a time order sequence of interval-of-validity
      The time associated to each interval is the end-validity (till time)
      the end of each interval is the begin of the next
      it is a UpdateStamp by mixin (I'll regret..)
   */
  class IOVSequence : public  UpdateStamp{
  public:
    typedef cond::IOVElement Item;
    typedef pool::PVector<Item> Container;
    typedef Container::iterator iterator;
    typedef Container::const_iterator const_iterator;

    IOVSequence();
    IOVSequence(int type, cond::Time_t since,std::string const& imetadata);

    ~IOVSequence();


    // append a new item, return new size
    size_t add(cond::Time_t time, 
	       std::string const & payloadToken,
	       std::string const & metadataToken);

    iterator find(cond::Time_t time);

    const_iterator find(cond::Time_t time) const;

    cond::TimeType timeType() const { return cond::timeTypeSpecs[m_timetype].type;}

    cond::Time_t lastTill() const { return  m_lastTill;}

    void updateLastTill(cond::Time_t since) { m_firstsince=since;}


    Container & iovs() { return m_iovs;}
    
    Container const & iovs() const  { return m_iovs;}

    // if true the sequence is not guaranted to be the same in previous version
    bool freeUpdate() const { return m_freeUpdate;}
    
    void enableFreeUpdate()  { m_freeUpdate=true;}


    std::string const & metadataToken() const { return m_metadata;}



  private:

    Container m_iovs;
    int m_timetype;
    cond::Time_t m_lastTill;

    bool m_freeUpdate;

    std::string m_metadata; // FIXME change in Pool::Ptr???

  };

}//ns cond
#endif
