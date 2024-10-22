#ifndef ODCOND2CONFINFO_H
#define ODCOND2CONFINFO_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int run_t;

class ODCond2ConfInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;
  ODCond2ConfInfo();
  ~ODCond2ConfInfo() override;

  // User data methods
  inline std::string getTable() override { return "COND_2_CONF_INFO"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setType(std::string x) { m_type = x; }
  inline std::string getType() const { return m_type; }

  inline void setRecordDate(const Tm& x) { m_rec_time = x; }
  inline Tm getRecordDate() const { return m_rec_time; }

  inline void setLocation(std::string x) { m_loc = x; }
  inline std::string getLocation() const { return m_loc; }

  inline void setRunNumber(int id) { m_run = id; }
  inline int getRunNumber() const { return m_run; }

  inline void setDescription(std::string x) { m_desc = x; }
  inline std::string getDescription() const { return m_desc; }

  inline void setDBDate(const Tm& x) { m_db_time = x; }
  inline Tm getDBDate() const { return m_db_time; }

  // the tag is already in IODConfig

  int fetchID() noexcept(false);

  int fetchNextId() noexcept(false);
  void setParameters(const std::map<std::string, std::string>& my_keys_map);

private:
  void prepareWrite() noexcept(false) override;

  void writeDB() noexcept(false);

  void clear();

  void fetchData(ODCond2ConfInfo* result) noexcept(false);

  void fetchParents() noexcept(false);

  // User data
  int m_ID;
  std::string m_type;
  Tm m_rec_time;
  std::string m_loc;
  int m_run;
  std::string m_desc;
  Tm m_db_time;
  int m_loc_id;
  int m_typ_id;
};

#endif
