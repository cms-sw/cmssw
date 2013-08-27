#ifndef FECONFIGBADStripINFO_H
#define FECONFIGBADStripINFO_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class FEConfigBadStripInfo : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  FEConfigBadStripInfo();
  ~FEConfigBadStripInfo();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_BadST_INFO"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  // the tag is already in IODConfig 

  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }
  int fetchID()  throw(std::runtime_error);

  int fetchNextId() throw(std::runtime_error);
  void setParameters(const std::map<std::string,std::string>& my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);

  void writeDB()       throw(std::runtime_error);

  void clear();

  void fetchData(FEConfigBadStripInfo * result)     throw(std::runtime_error);



  // User data
  int m_ID;
  int m_version;
  
};

#endif
