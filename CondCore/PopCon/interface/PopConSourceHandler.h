#ifndef  PopConSourceHandler_H
#define  PopConSourceHandler_H

//#include "CondCore/DBCommon/interface/DbSession.h"
//#include "CondCore/DBCommon/interface/DbTransaction.h"

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Time.h"
//#include "CondCore/DBCommon/interface/TagInfo.h"
//#include "CondCore/DBCommon/interface/LogDBEntry.h"

#include <boost/bind.hpp>
#include <algorithm>
#include <vector>
#include <string>

namespace cond {
  class Summary;
}

#include "CondFormats/Common/interface/GenericSummary.h"



namespace popcon {

  /** Online DB source handler, aims at returning the vector of data to be 
   * transferred to the online database
   * Subdetector developers inherit over this class with template parameter of 
   * payload class; 
   * need just to implement the getNewObjects method that loads the calibs,
   * the sourceId methods that return a text identifier of the source,
   * and provide a constructor that accept a ParameterSet
   */
  template <class T>
  class PopConSourceHandler{
  public: 
    typedef T value_type;
    typedef PopConSourceHandler<T> self;
    typedef cond::Time_t Time_t;
    typedef cond::Summary Summary;
    
    struct Triplet {
      value_type * payload;
      Summary * summary;
      Time_t time;
    };
    
    typedef std::vector<Triplet> Container;
    
    typedef std::vector<std::pair<T*, cond::Time_t> > OldContainer;
    
    
    class Ref {
    public:
      Ref() : m_dbsession(){}
      Ref(cond::persistency::Session& dbsession, const std::string& hash) : 
        m_dbsession(dbsession){
	m_d = m_dbsession.fetchPayload<T>( hash );
      }
      ~Ref() {
      }
      
      Ref(const Ref & ref) :
        m_dbsession(ref.m_dbsession), m_d(ref.m_d) {
      }
      
      Ref & operator=(const Ref & ref) {
        m_dbsession = ref.m_dbsession;
        m_d = ref.m_d;
        return *this;
      }
      
      T const * ptr() const {
        return m_d.get();
      }
      
      T const * operator->() const {
        return ptr();
      }
      // dereference operator
      T const & operator*() const {
        return *ptr();
      }
      
      
    private:
      
      cond::persistency::Session m_dbsession;
      boost::shared_ptr<T> m_d;
    };
    
    
    PopConSourceHandler():
      m_tagInfo(0),
      m_logDBEntry(0)
    {}
    
    virtual ~PopConSourceHandler(){
    }
    
    
    cond::TagInfo_t const & tagInfo() const { return  *m_tagInfo; }
    
    // return last paylod of the tag
    Ref lastPayload() const {
      return Ref(m_session,tagInfo().lastPayloadToken);
    }
    
    // return last successful log entry for the tag in question
    cond::LogDBEntry_t const & logDBEntry() const { return *m_logDBEntry; }
    
    // FIX ME
    void initialize (const cond::persistency::Session& dbSession,
      		     cond::TagInfo_t const & tagInfo, cond::LogDBEntry_t const & logDBEntry) { 
      m_session = dbSession;
      m_tagInfo = &tagInfo;
      m_logDBEntry = &logDBEntry;
    }
    
    // this is the only mandatory interface
    std::pair<Container const *, std::string const>  operator()(const cond::persistency::Session& session,
      							cond::TagInfo_t const & tagInfo, 
      							cond::LogDBEntry_t const & logDBEntry) const {
      const_cast<self*>(this)->initialize(session, tagInfo, logDBEntry);
      return std::pair<Container const *, std::string const>(&(const_cast<self*>(this)->returnData()), userTextLog());
    }
    
    Container const &  returnData() {
      getNewObjects();
      if (!m_to_transfer.empty()) convertFromOld();
      sort();
      return m_triplets;
    }
    
    std::string const & userTextLog() const { return m_userTextLog; }
    
    //Implement to fill m_to_transfer vector and  m_userTextLog
    //use getOfflineInfo to get the contents of offline DB
    virtual void getNewObjects()=0;
    
    // return a string identifing the source
    virtual std::string id() const=0;
    
    void sort() {
      std::sort(m_triplets.begin(),m_triplets.end(),
		boost::bind(std::less<cond::Time_t>(),
			    boost::bind(&Container::value_type::time,_1),
			    boost::bind(&Container::value_type::time,_2)
			    )
		);
    }
    
    
    // make sure to create a new one each time...
    Summary * dummySummary(typename OldContainer::value_type const &) const {
      return new cond::GenericSummary("not supplied");
    }
    
    void convertFromOld() {
      std::for_each( m_to_transfer.begin(), m_to_transfer.end(),
		     boost::bind(&self::add, this,
				 boost::bind(&OldContainer::value_type::first,_1),
				 boost::bind(&self::dummySummary, this, _1),
				 boost::bind(&OldContainer::value_type::second,_1)
				 ));
    }
    
  protected:
    
    
    int add(value_type * payload, Summary * summary, Time_t time) {
      Triplet t = {payload,summary,time};
      m_triplets.push_back(t);
      return m_triplets.size();
    }

  private:
    
    mutable cond::persistency::Session m_session;
    
    cond::TagInfo_t const * m_tagInfo;
    
    cond::LogDBEntry_t const * m_logDBEntry;
    

  protected:
    
    //vector of payload objects and iovinfo to be transferred
    //class looses ownership of payload object
    OldContainer m_to_transfer;

    private:
    Container m_triplets;

  protected:
    std::string m_userTextLog;


  };
}
#endif
