#ifndef RUNTAG_H
#define RUNTAG_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/LocationDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
/**
 *   Tag for Run information
 */
class RunTag : public ITag {
 public:
  friend class RunIOV;  // needs permission to write
  friend class EcalCondDBInterface;

  RunTag();
  ~RunTag();

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  LocationDef getLocationDef() const;
  void setLocationDef(const LocationDef locDef);

  RunTypeDef getRunTypeDef() const;
  void setRunTypeDef(const RunTypeDef runTypeDef);

  // Methods using ID
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const RunTag& t) const
    {
      return (m_genTag == t.m_genTag &&
	      m_locDef == t.m_locDef &&
	      m_runTypeDef == t.m_runTypeDef);
    }
  
  inline bool operator!=(const RunTag& t) const { return !(*this == t); }

 private:
  // User data for this tag
  std::string m_genTag;
  LocationDef m_locDef;
  RunTypeDef m_runTypeDef;

  // Methods from ITag
  int writeDB() throw(std::runtime_error);
  void fetchParentIDs(int* locId, int* runTypeID) throw(std::runtime_error);

  // Public access methods
  void fetchAllTags( std::vector<RunTag>* fillVec) throw(std::runtime_error);

};

#endif
