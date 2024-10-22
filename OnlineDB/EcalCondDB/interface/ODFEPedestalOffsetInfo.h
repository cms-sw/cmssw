#ifndef ODFEPEDOFFINFO_H
#define ODFEPEDOFFINFO_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODFEPedestalOffsetInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;
  ODFEPedestalOffsetInfo();
  ~ODFEPedestalOffsetInfo() override;

  // User data methods
  inline std::string getTable() override { return "PEDESTAL_OFFSETS_INFO"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  // the tag is already in IODConfig

  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }

  int fetchNextId() noexcept(false);
  void setParameters(const std::map<std::string, std::string>& my_keys_map);
  int fetchID() noexcept(false);

private:
  void prepareWrite() noexcept(false) override;

  void writeDB() noexcept(false);

  void clear();

  void fetchData(ODFEPedestalOffsetInfo* result) noexcept(false);
  void fetchLastData(ODFEPedestalOffsetInfo* result) noexcept(false);

  // User data
  int m_ID;
  int m_version;
};

#endif
