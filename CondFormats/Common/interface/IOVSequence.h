#ifndef Cond_IOVSequence_h
#define Cond_IOVSequence_h
#include "CondFormats/Common/interface/UpdateStamp.h"
#include "CondFormats/Common/interface/IOVElement.h"
#include "CondFormats/Common/interface/Time.h"
#include <vector>
#include <string>
#include "POOLCore/PVector.h"
#include "POOLCore/Ptr.h"

#include "CondFormats/Common/interface/IOVProvenance.h"
#include "CondFormats/Common/interface/IOVDescription.h"
#include "CondFormats/Common/interface/IOVUserMetaData.h"


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

    // the real default constructor...
    explicit IOVSequence(cond::TimeType ttype);

    // constructor for the editor
    IOVSequence(int type, cond::Time_t till, std::string const& imetadata);

    ~IOVSequence();

    IOVSequence(IOVSequence const & rh);
    IOVSequence & operator=(IOVSequence const & rh);

    // append a new item, return position of last inserted entry
    size_t add(cond::Time_t time, 
	       std::string const & wrapperToken);

    // remove last entry, return position of last entry still valid
    size_t truncate();


    // find IOV for which time is valid (this is not STANDARD std::find!)
    const_iterator find(cond::Time_t time) const;

    // true if an iov with since==time already exists
    bool exist(cond::Time_t time) const;

    cond::TimeType timeType() const { return cond::timeTypeSpecs[m_timetype].type;}

    // FIXME shall we cache it?
    cond::Time_t firstSince() const { return  iovs().front().sinceTime();}

    cond::Time_t lastTill() const { return  m_lastTill;}

    void updateLastTill(cond::Time_t till) { m_lastTill=till;}

    
  public:
    Container const & iovs() const;

    // if true the "sorted" sequence is not guaranted to be the same as in previous version
    bool notOrdered() const { return m_notOrdered;}
    
    std::string const & metadataToken() const { return m_metadata;}


    // void set_description(cond::IOVDescription * id) { m_description = id;}

    // cond::IOVDescription const & description() const {return *m_description;}


    void loadAll() const;

  public:

    // the real persistent container...
    Container & piovs() { return m_iovs;}
    Container const & piovs() const { return m_iovs;}

  private:
    
    // iovs is not in order: take action!
    void disorder();

    // sort the container in m_sorted
    Container const & sortMe() const;
    
  private:

    Container m_iovs;
    int m_timetype;
    cond::Time_t m_lastTill;

    bool m_notOrdered;

    std::string m_metadata; // FIXME not used???

    // to describe history and be used as provenance
    // pool::PolyPtr<cond::IOVProvenance> m_provenance;

    // keep list of types, list of keywords, 
    // pool::PolyPtr<cond::IOVDescription> m_description;

    // for the user
    // pool::PolyPtr<cond::IOVUserMetaData> m_userMetadata;

    mutable Container * m_sorted;

  };

}//ns cond
#endif
