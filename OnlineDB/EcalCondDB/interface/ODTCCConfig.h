#ifndef ODTCCCONFIG_H
#define ODTCCCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODTCCConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODTCCConfig();
  ~ODTCCConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_TCC_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setTCCConfigurationFile(std::string x) { m_tcc_file = x; }
  inline std::string getTCCConfigurationFile() const { return m_tcc_file; }
  inline void setLUTConfigurationFile(std::string x) { m_lut_file = x; }
  inline std::string getLUTConfigurationFile() const { return m_lut_file; }
  inline void setSLBConfigurationFile(std::string x) { m_slb_file = x; }
  inline std::string getSLBConfigurationFile() const { return m_slb_file; }
  inline void setTestPatternFileUrl(std::string x) { m_test_url = x; }
  inline std::string getTestPatternFileUrl() const { return m_test_url; }
  inline void setNTestPatternsToLoad(int id) { m_ntest = id; }
  inline int getNTestPatternsToLoad() const { return m_ntest; }

  inline void setTCCClob(unsigned char* x) { m_tcc_clob = x; }
  inline unsigned char* getTCCClob() const { return m_tcc_clob; }

  inline void setLUTClob(unsigned char* x) { m_lut_clob = x; }
  inline unsigned char* getLUTClob() const { return m_lut_clob; }

  inline void setSLBClob(unsigned char* x) { m_slb_clob = x; }
  inline unsigned char* getSLBClob() const { return m_slb_clob; }

  void setParameters(std::map<std::string,std::string> my_keys_map);

  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODTCCConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_tcc_clob;
  unsigned char* m_lut_clob;
  unsigned char* m_slb_clob;
  std::string  m_tcc_file;
  std::string  m_lut_file;
  std::string  m_slb_file;
  std::string  m_test_url;
  int  m_ntest;
  unsigned int m_size;
};

#endif
