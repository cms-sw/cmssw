#ifndef MONRUNOUTCOMEDEF_H
#define MONRUNOUTCOMEDEF_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDef.h"

/**
 *   Def for monitoring run outcomes
 */
class MonRunOutcomeDef : public IDef {
  public:
  friend class EcalCondDBInterface;
  
  MonRunOutcomeDef();
  virtual ~MonRunOutcomeDef();

  // Methods for user data
  std::string getShortDesc() const;
  void setShortDesc(std::string desc);

  std::string getLongDesc() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators
  inline bool operator==(const MonRunOutcomeDef &d) const { return m_shortDesc == d.m_shortDesc; }
  inline bool operator!=(const MonRunOutcomeDef &d) const { return m_shortDesc != d.m_shortDesc; }

 protected:
  // User data for this def
  std::string m_shortDesc;
  std::string m_longDesc;

  void fetchAllDefs( std::vector<MonRunOutcomeDef>* fillVec) throw(std::runtime_error);
};

#endif
