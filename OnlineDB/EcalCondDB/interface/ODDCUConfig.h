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
  ~ODDCUConfig() override;

  // User data methods
  inline std::string getTable() override { return "ECAL_DCU_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  void setParameters(const std::map<std::string, std::string>& my_keys_map);

private:
  void prepareWrite() noexcept(false) override;
  void writeDB() noexcept(false);
  void clear();
  void fetchData(ODDCUConfig* result) noexcept(false);
  int fetchID() noexcept(false);

  int fetchNextId() noexcept(false);

  // User data
  int m_ID;
};

#endif
