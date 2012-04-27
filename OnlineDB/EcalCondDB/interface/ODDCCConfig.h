#ifndef ODDCCCONFIG_H
#define ODDCCCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODDCCConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODDCCConfig();
  ~ODDCCConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_DCC_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setDCCConfigurationUrl(std::string x) { m_dcc_url = x; }
  inline std::string getDCCConfigurationUrl() const { return m_dcc_url; }

  inline void setTestPatternFileUrl(std::string x) { m_test_url = x; }
  inline std::string getTestPatternFileUrl() const { return m_test_url; }

  inline void setNTestPatternsToLoad(int id) { m_ntest = id; }
  inline int getNTestPatternsToLoad() const { return m_ntest; }

   inline void setSMHalf(int id) { m_sm_half = id; }
   inline int getSMHalf() const { return m_sm_half; }

  inline void setDCCClob(unsigned char* x) { m_dcc_clob = x; }
  inline unsigned char* getDCCClob() const { return m_dcc_clob; }
  inline unsigned int getDCCClobSize() const { return m_size; }
  inline void setDCCWeightsMode(std::string x) { m_wei = x; }
  inline std::string getDCCWeightsMode() const { return m_wei; }

  void setParameters(std::map<std::string,std::string> my_keys_map);

  inline void printout() { 
    std::cout <<"DCC >>" << "SIZE " <<  getSize()<< std::endl;
    std::cout <<"DCC >>" << "DCCConfigurationUrl " <<  getDCCConfigurationUrl() << std::endl;
    std::cout <<"DCC >>" << "TestPatternFileUrl " <<  getTestPatternFileUrl()<< std::endl;
    std::cout <<"DCC >>" << "NTestPatternsToLoad " <<  getNTestPatternsToLoad()<< std::endl;
    std::cout <<"DCC >>" << "SMHalf " <<  getSMHalf()<< std::endl;
    std::cout <<"DCC >>" << "DCCWeightsMode" <<  getDCCWeightsMode()<< std::endl;
    
  }

  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODDCCConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_dcc_clob;
  std::string  m_dcc_url;
  std::string  m_test_url;
  int  m_ntest;
  int  m_sm_half;
  unsigned int m_size; 
  std::string  m_wei;

};

#endif
