#ifndef TEMPLATE_PAYLOAD_H
#define TEMPLATE_PAYLOAD_H


#include "CondCore/PopCon/interface/IOVPair.h"
#include "CondCore/PopCon/interface/OfflineDBInterface.h"

#include <vector>
#include <string>

namespace popcon {

  //Online DB source handler, aims at returning the vector of data to be 
  //transferred to the online database
  //Subdetector developers inherit over this class with template parameter of 
  //payload class; need to implement the getNewObjects method and overload the 
  //constructor; 
  
  template <class T>
    class PopConSourceHandler{
    public: 

    typedef PopConSourceHandler<T> self;

    PopConSourceHandler(const std::string& connect_string) : 
      m_db_iface(connect_string) {}

 
    virtual ~PopConSourceHandler(){
    }
    
    // this is the only mandatory interface
    std::vector<std::pair<T*, popcon::IOVPair> > const & operator()() const {
      return const_cast<self*>(this)->returnData();
    }

    Time_t getSinceForTag(const std::string& tag) const {
      return (m_db_iface.getSpecificTagInfo(tag)).last_since;
    }

    std::vector<std::pair<T*, popcon::IOVPair> > const &  returnData() {
      this->getNewObjects();
      return this->m_to_transfer;
    }
    
    //Implement to fill m_to_transfer vector
    //use getOfflineInfo to get the contents of offline DB
    virtual void getNewObjects()=0;

    // return a string identifing the source
    virtual std::string id() const=0;
   
    private:
    //Offline Database Interface object
    popcon::OfflineDBInterface m_db_iface;
    
    protected:
     
    //Is is sufficient for getNewObjects algorithm?
    std::map<std::string, PayloadIOV> getOfflineInfo(){
      return m_db_iface.getStatusMap();
    }
      
    //vector of payload objects and iovinfo to be transferred
    //class looses ownership of payload object
    std::vector<std::pair<T*, popcon::IOVPair> >  m_to_transfer;
  };
}
#endif
