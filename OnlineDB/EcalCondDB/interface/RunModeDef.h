#ifndef RUNMODEDEF_H
#define RUNMODEDEF_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDef.h"

/**
 *   Def for Location information
 */
class RunModeDef : public IDef {
  public:
  friend class EcalCondDBInterface;
  
  RunModeDef();
  virtual ~RunModeDef();

  // Methods for user data
  std::string getRunMode() const;
  void setRunMode(std::string runmode);
  


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators.  m_desc is not considered, it cannot be written to DB anyhow
  inline bool operator==(const RunModeDef &t) const { return m_runMode == t.m_runMode; }
  inline bool operator!=(const RunModeDef &t) const { return m_runMode != t.m_runMode; }
  
 protected:
  // User data for this def
  std::string m_runMode;


  void fetchAllDefs( std::vector<RunModeDef>* fillVec) throw(std::runtime_error);
};

#endif
