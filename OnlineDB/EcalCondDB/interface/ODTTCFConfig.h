#ifndef ODTTCFCONFIG_H
#define ODTTCFCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODTTCFConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODTTCFConfig();
  ~ODTTCFConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_TTCF_CONFIGURATION"; }
  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setTTCFConfigurationFile(std::string x) { m_ttcf_file = x; }
  inline std::string getTTCFConfigurationFile() const { return m_ttcf_file; }

  inline void setTTCFClob(unsigned char* x) { m_ttcf_clob = x; }
  inline unsigned char* getTTCFClob() const { return m_ttcf_clob; }

  inline void setRxBC0Delay( int x ) { m_rxbc0_delay = x; } 
  inline int getRxBC0Delay( ) { return m_rxbc0_delay; } 
  inline void setReg30( int x ) { m_reg_30 = x; } 
  inline int getReg30( ) { return m_reg_30; } 

  void setParameters(std::map<std::string,std::string> my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);

  void writeDB()       throw(std::runtime_error);

  void clear();

  void fetchData(ODTTCFConfig * result)     throw(std::runtime_error);

  int fetchID()  throw(std::runtime_error);



  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_ttcf_clob;
  unsigned int m_size;
  std::string m_ttcf_file;

  int m_rxbc0_delay;
  int m_reg_30;
  
};

#endif
