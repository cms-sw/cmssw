#ifndef CALITAG_H
#define CALITAG_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/LocationDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
/**
 *   Tag for Run information
 */
class CaliTag : public ITag {
 public:
  friend class CaliIOV;  // needs permission to write
  friend class EcalCondDBInterface;

  CaliTag();
  ~CaliTag();

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  LocationDef getLocationDef() const;
  void setLocationDef(const LocationDef locDef);

  std::string getMethod() const;
  void setMethod(std::string method);

  std::string getVersion() const;
  void setVersion(std::string version);
  
  std::string getDataType() const;
  void setDataType(std::string dataType);


  // Methods using ID
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators
  inline bool operator==(const CaliTag &t) const 
    { 
      return (m_genTag == t.m_genTag &&
	      m_locDef == t.m_locDef);
    }

  inline bool operator!=(const CaliTag &t) const { return !(*this == t); }

 private:
  // User data for this tag
  std::string m_genTag;
  LocationDef m_locDef;
  std::string m_method;
  std::string m_version;
  std::string m_dataType;

  // Methods from ITag
  int writeDB() throw(std::runtime_error);
  void fetchParentIDs(int* locId) throw(std::runtime_error);

  // Public access methods
  void fetchAllTags( std::vector<CaliTag>* fillVec) throw(std::runtime_error);

};

#endif
