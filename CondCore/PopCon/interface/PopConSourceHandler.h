#ifndef  PopConSourceHandler_H
#define  PopConSourceHandler_H

#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/TagInfo.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"

#include <boost/bind.hpp>
#include <algorithm>
#include <vector>
#include <string>

namespace cond {
  class Summary;
}

#include "CondFormats/Common/interface/PayloadWrapper.h"
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
    typedef cond::DataWrapper<value_type> Wrapper;
    
    struct Triplet {
      value_type * payload;
      Summary * summary;
      Time_t time;
    };
    
    typedef std::vector<Triplet> Container;
    
    typedef std::vector<std::pair<T*, cond::Time_t> > OldContainer;
    
    
    class Ref : public cond::TypedRef<Wrapper>  {
    public:
      Ref() : m_pooldb(0){}
      Ref(cond::PoolTransaction& pooldb, std::string token) : 
        m_pooldb(&pooldb){
	m_pooldb->start(true);
	(cond::TypedRef<Wrapper>&)(*this) = cond::TypedRef<Wrapper>(pooldb,token);
      }
      ~Ref() {
	if (m_pooldb)
	  m_pooldb->commit();
      }
      
      Ref(const Ref & ref) : 
	cond::TypedRef<Wrapper>(ref), m_pooldb(ref.m_pooldb) {
	ref.m_pooldb=0; // avoid commit;
      }
      
      Ref & operator=(const Ref & ref) {
	cond::TypedRef<Wrapper>::operator=(ref);
	m_pooldb = ref.m_pooldb;
	ref.m_pooldb=0; // avoid commit;
	return *this;
      }
      
      mutable cond::PoolTransaction *m_pooldb;
      
    };
    
    
    PopConSourceHandler(){}
    
    virtual ~PopConSourceHandler(){
    }
    
    
    cond::TagInfo const & tagInfo() const { return  *m_tagInfo; }
    
    // return last paylod of the tag
    Ref lastPayload() const {
      return Ref(m_connection->poolTransaction(),tagInfo().lastPayloadToken);
    }
    
    // return last successful log entry for the tag in question
    cond::LogDBEntry const & logDBEntry() const { return *m_logDBEntry; }
    
    
    void initialize (cond::Connection* connection,
		     cond::TagInfo const & tagInfo, cond::LogDBEntry const & logDBEntry) { 
      m_connection = connection;
      m_tagInfo = &tagInfo;
      m_logDBEntry = &logDBEntry;
    }
    
    // this is the only mandatory interface
    std::pair<Container const *, std::string const>  operator()(cond::Connection* connection,
								cond::TagInfo const & tagInfo, 
								cond::LogDBEntry const & logDBEntry) const {
      const_cast<self*>(this)->initialize(connection, tagInfo, logDBEntry);
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
    
    
    
    void convertFromOld() {
     std::for_each( m_to_transfer.begin(), m_to_transfer.end(),
		   boost::bind(&self::add, this,
			       boost::bind(&OldContainer::value_type::first,_1),
			       new cond::GenericSummary("not supplied"),
			       boost::bind(&OldContainer::value_type::second,_1)
			       ));
    }

  protected:


    int add(value_type * payload, Summary * summary, Time_t time) {
      m_triplets.push_back({payload,summary,time});
    }

  private:
    
    cond::Connection* m_connection;
    
    cond::TagInfo const * m_tagInfo;
    
    cond::LogDBEntry const * m_logDBEntry;
    

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
