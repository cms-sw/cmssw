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
    typedef std::vector<std::pair<T*, cond::Time_t> > Container;
    typedef cond::Time_t Time_t;

    class Ref : public cond::TypedRef<value_type>  {
    public:
      Ref() : m_pooldb(0){}
      Ref(cond::PoolTransaction& pooldb, std::string token) : 
        m_pooldb(&pooldb){
	m_pooldb->start(true);
	(cond::TypedRef<value_type>&)(*this) = cond::TypedRef<value_type>(pooldb,token);
      }
      ~Ref() {
	if (m_pooldb)
	  m_pooldb->commit();
      }

      Ref(const Ref & ref) : 
	cond::TypedRef<value_type>(ref), m_pooldb(ref.m_pooldb) {
	ref.m_pooldb=0; // avoid commit;
      }

      Ref & operator=(const Ref & ref) {
	cond::TypedRef<value_type>::operator=(ref);
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
      sort();
      return m_to_transfer;
    }
    
    std::string const & userTextLog() const { return m_userTextLog; }


    //Implement to fill m_to_transfer vector and  m_userTextLog
    //use getOfflineInfo to get the contents of offline DB
    virtual void getNewObjects()=0;

    // return a string identifing the source
    virtual std::string id() const=0;

    void sort() {
      std::sort(m_to_transfer.begin(),m_to_transfer.end(),
		boost::bind(std::less<cond::Time_t>(),
			    boost::bind(&Container::value_type::second,_1),
			    boost::bind(&Container::value_type::second,_2)
			    )
		);
    }


    
  private:
    
    cond::Connection* m_connection;

    cond::TagInfo const * m_tagInfo;
    
    cond::LogDBEntry const * m_logDBEntry;
    

  protected:
    
    //vector of payload objects and iovinfo to be transferred
    //class looses ownership of payload object
    Container m_to_transfer;

    std::string m_userTextLog;


  };
}
#endif
