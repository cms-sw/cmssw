#ifndef ODTTCCICONFIG_H
#define ODTTCCICONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODTTCciConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODTTCciConfig();
  ~ODTTCciConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_TTCci_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setTTCciConfigurationFile(std::string x) { m_ttcci_file = x; }
  inline std::string getTTCciConfigurationFile() const { return m_ttcci_file; }

  inline void setTrgMode(std::string x) { m_trg_mode = x; }
  inline std::string getTrgMode() const { return m_trg_mode; }
  inline void setTrgSleep(int id) { m_trg_sleep = id; }
  inline int getTrgSleep() const { return m_trg_sleep; }

  inline void setTTCciClob(unsigned char* x) { m_ttcci_clob = x; }
  inline unsigned char* getTTCciClob() const { return m_ttcci_clob; }

  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODTTCciConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);

  unsigned char* readClob (oracle::occi::Clob &clob, int size)   throw(std::runtime_error);
  void populateClob (oracle::occi::Clob &clob)  throw(std::runtime_error);
  void dumpClob (oracle::occi::Clob &clob,unsigned int way)   throw (std::runtime_error);

  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_ttcci_clob;
  std::string  m_ttcci_file;
  std::string  m_trg_mode;
  int  m_trg_sleep;

  
};

#endif
