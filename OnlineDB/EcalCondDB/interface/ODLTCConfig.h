#ifndef ODLTCCONFIG_H
#define ODLTCCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODLTCConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODLTCConfig();
  ~ODLTCConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_LTC_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setLTCConfigurationFile(std::string x) { m_ltc_file = x; }
  inline std::string getLTCConfigurationFile() const { return m_ltc_file; }

  inline void setLTCClob(unsigned char* x) { m_ltc_clob = x; }
  inline unsigned char* getLTCClob() const { return m_ltc_clob; }

  void setParameters(std::map<std::string,std::string> my_keys_map);

  inline void printout(){ 
    std::cout <<"LTC >>" << "LTCConfigurationFile " <<  getLTCConfigurationFile()<< std::endl;
     
  }

  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODLTCConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_ltc_clob;
  std::string  m_ltc_file;
  int m_size;

};

#endif
