#ifndef ODFEDAQCONFIG_H
#define ODFEDAQCONFIG_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODFEDAQConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODFEDAQConfig();
  ~ODFEDAQConfig();

  // User data methods
  inline std::string getTable() { return "FE_DAQ_CONFIG"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  // the tag is already in IODConfig 

  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }

  inline void setPedestalId(int x) { m_ped = x; }
  inline int getPedestalId() const { return m_ped; }
  inline void setDelayId(int x) { m_del = x; }
  inline int getDelayId() const { return m_del; }
  inline void setWeightId(int x) { m_wei = x; }
  inline int getWeightId() const { return m_wei; }

  inline void setBadXtId(int x) { m_bxt = x; }
  inline int getBadXtId() const { return m_bxt; }
  inline void setBadTTId(int x) { m_btt = x; }
  inline int getBadTTId() const { return m_btt; }
  inline void setTriggerBadXtId(int x) { m_tbxt = x; }
  inline int getTriggerBadXtId() const { return m_tbxt; }
  inline void setTriggerBadTTId(int x) { m_tbtt = x; }
  inline int getTriggerBadTTId() const { return m_tbtt; }

  inline void setComment(std::string x) { m_com = x; }
  inline std::string getComment() const { return m_com; }

  int fetchNextId() throw(std::runtime_error);
  void setParameters(std::map<std::string,std::string> my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);

  void writeDB()       throw(std::runtime_error);

  void clear();

  void fetchData(ODFEDAQConfig * result)     throw(std::runtime_error);

  int fetchID()  throw(std::runtime_error);


  // User data
  int m_ID;
  int m_ped;
  int m_del;
  int m_wei;

  int m_bxt;
  int m_btt;
  int m_tbxt;
  int m_tbtt;
  int m_version;
  std::string m_com;
  
};

#endif
