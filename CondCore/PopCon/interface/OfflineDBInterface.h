#ifndef OFFLINE_DB_INTERFACE_H
#define OFFLINE_DB_INTERFACE_H


#include "CondCore/PopCon/interface/IOVPair.h"

#include <string>
#include <map>

namespace popcon
{
  //mapped type for subdetector information on offline db contents 
  struct PayloadIOV{
    size_t number_of_objects;
    //last payload object IOV info 
    Time_t last_since;
    Time_t last_till;
    std::string container_name;	
  };
  
  class OfflineDBInterface{
  public:
    typedef std::map<std::string, PayloadIOV> States; 
    OfflineDBInterface(const std::string& connect);
    virtual ~OfflineDBInterface();
    virtual States const & getStatusMap() const ;
    PayloadIOV getSpecificTagInfo(const std::string& tag) const;
  private:
     //tag - IOV/Payload information map
    mutable States m_status_map;
    std::string m_connect;
    void getAllTagsInfo() const;
    void getSpecificPayloadMap(const std::string& ) const;
  }; 
}
#endif
