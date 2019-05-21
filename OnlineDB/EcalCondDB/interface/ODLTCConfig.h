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
  ~ODLTCConfig() override;

  // User data methods
  inline std::string getTable() override { return "ECAL_LTC_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setLTCConfigurationFile(std::string x) { m_ltc_file = x; }
  inline std::string getLTCConfigurationFile() const { return m_ltc_file; }

  inline void setLTCClob(unsigned char* x) { m_ltc_clob = x; }
  inline unsigned char* getLTCClob() const { return m_ltc_clob; }

  void setParameters(const std::map<std::string, std::string>& my_keys_map);

private:
  void prepareWrite() noexcept(false) override;
  void writeDB() noexcept(false);
  void clear();
  void fetchData(ODLTCConfig* result) noexcept(false);
  int fetchID() noexcept(false);

  int fetchNextId() noexcept(false);

  // User data
  int m_ID;
  unsigned char* m_ltc_clob;
  std::string m_ltc_file;
  int m_size;
};

#endif
