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
  virtual ~RunTypeDef();

  // Methods for user data
  std::string getRunType() const;
  void setRunType(std::string runtype);
  
  std::string getConfigTag() const;
  void setConfigTag(std::string tag);

  int getConfigVersion() const;
  void setConfigVersion(int ver);

  std::string getDescription() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

 protected:
  // User data for this def
  std::string m_runType;
  std::string m_configTag;
  int m_configVer;
  std::string m_desc;

  void fetchAllDefs( std::vector<RunTypeDef>* fillVec) throw(std::runtime_error);
};

#endif
