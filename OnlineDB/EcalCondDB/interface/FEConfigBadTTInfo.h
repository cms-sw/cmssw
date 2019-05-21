#ifndef FECONFIGBADTTINFO_H
#define FECONFIGBADTTINFO_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class FEConfigBadTTInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;
  FEConfigBadTTInfo();
  ~FEConfigBadTTInfo() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_BADTT_INFO"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  // the tag is already in IODConfig

  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }
  int fetchID() noexcept(false);

  int fetchNextId() noexcept(false);
  void setParameters(const std::map<std::string, std::string>& my_keys_map);

private:
  void prepareWrite() noexcept(false) override;

  void writeDB() noexcept(false);

  void clear();

  void fetchData(FEConfigBadTTInfo* result) noexcept(false);

  // User data
  int m_ID;
  int m_version;
};

#endif
