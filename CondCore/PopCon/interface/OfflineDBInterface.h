#ifndef OFFLINE_DB_INTERFACE_H
#define OFFLINE_DB_INTERFACE_H

#include <string>
#include <map>
namespace cond{
  class DBSession;
}
namespace popcon
{
  //mapped type for subdetector information on offline db contents 
  struct PayloadIOV{
    unsigned int number_of_objects;
    //last payload object IOV info 
    unsigned int last_since;
    unsigned int last_till;
    std::string container_name;	
  };
  
  class OfflineDBInterface{
  public:	
    OfflineDBInterface(const std::string& connect);
    virtual ~OfflineDBInterface();
    virtual std::map<std::string, PayloadIOV> getStatusMap();
    PayloadIOV getSpecificTagInfo(const std::string& tag);
  private:
    //tag - IOV/Payload information map
    std::map<std::string, PayloadIOV> m_status_map;
    std::string m_connect;
    //std::string m_user;
    //std::string m_pass;
    cond::DBSession* session;  
    void getAllTagsInfo();
    void getSpecificPayloadMap(const std::string& );
  }; 
}
#endif
