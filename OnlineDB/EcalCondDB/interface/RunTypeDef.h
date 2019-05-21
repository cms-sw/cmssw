#ifndef RUNTYPEDEF_H
#define RUNTYPEDEF_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDef.h"

/**
 *   Def for Location information
 */
class RunTypeDef : public IDef {
public:
  friend class EcalCondDBInterface;

  RunTypeDef();
  ~RunTypeDef() override;

  // Methods for user data
  std::string getRunType() const;
  void setRunType(std::string runtype);

  std::string getDescription() const;

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators.  m_desc is not considered, it cannot be written to DB anyhow
  inline bool operator==(const RunTypeDef &t) const { return m_runType == t.m_runType; }
  inline bool operator!=(const RunTypeDef &t) const { return m_runType != t.m_runType; }

protected:
  // User data for this def
  std::string m_runType;
  std::string m_desc;

  void fetchAllDefs(std::vector<RunTypeDef> *fillVec) noexcept(false);
};

#endif
