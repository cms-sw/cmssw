#ifndef ODSRPCONFIG_H
#define ODSRPCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;


class ODSRPConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODSRPConfig();
  ~ODSRPConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_SRP_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setDebugMode(int x) { m_debug = x; }
  inline int getDebugMode() const { return m_debug; }

  inline void setDummyMode(int x) { m_dummy= x; }
  inline int getDummyMode() const { return m_dummy; }

  inline void setPatternDirectory(std::string x) { m_patdir = x; }
  inline std::string getPatternDirectory() const { return m_patdir; }

  inline void setAutomaticMasks(int x) { m_auto = x; }
  inline int getAutomaticMasks() const { return m_auto; }

  inline void setAutomaticSrpSelect(int x) { m_auto_srp = x; }
  inline int getAutomaticSrpSelect() const { return m_auto_srp; }

  inline void setSRP0BunchAdjustPosition(int x) { m_bnch = x; }
  inline int getSRP0BunchAdjustPosition() const { return m_bnch; }

  inline void setConfigFile(std::string x) { m_file = x; }
  inline std::string getConfigFile() const { return m_file; }

  inline void setSRPClob(unsigned char* x) { m_srp_clob = x; }
  inline unsigned char* getSRPClob() const { return m_srp_clob; }
  inline unsigned int getSRPClobSize() const { return m_size; }

  void setParameters(std::map<std::string,std::string> my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODSRPConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  int fetchNextId() throw(std::runtime_error);

  // User data
  int m_ID;
  unsigned char* m_srp_clob;
  int m_debug;
  int m_dummy;
  std::string m_file;
  std::string m_patdir;
  int m_auto, m_auto_srp;
  int m_bnch;
  unsigned int m_size;

};

#endif
