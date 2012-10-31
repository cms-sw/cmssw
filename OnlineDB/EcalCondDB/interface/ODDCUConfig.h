#ifndef ODDCUCONFIG_H
#define ODDCUCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODDCUConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODDCUConfig();
  ~ODDCUConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_DCU_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  void setParameters(std::map<std::string,std::string> my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODDCUConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  
};

#endif
