#ifndef MONVERSIONDEF_H
#define MONVERSIONDEF_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDef.h"

/**
 *   Def for Location information
 */
class MonVersionDef : public IDef {
  public:
  friend class EcalCondDBInterface;
  
  MonVersionDef();
  virtual ~MonVersionDef();

  // Methods for user data
  std::string getMonitoringVersion() const;
  void setMonitoringVersion(std::string ver);

  std::string getDescription() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators
  inline bool operator==(const MonVersionDef &d) const { return m_monVer == d.m_monVer; }
  inline bool operator!=(const MonVersionDef &d) const { return m_monVer != d.m_monVer; }

 protected:
  // User data for this def
  std::string m_monVer;
  std::string m_desc;

  void fetchAllDefs( std::vector<MonVersionDef>* fillVec) throw(std::runtime_error);
};

#endif
