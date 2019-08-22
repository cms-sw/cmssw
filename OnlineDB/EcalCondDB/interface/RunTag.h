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
  ~RunTag() override;

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  LocationDef getLocationDef() const;
  void setLocationDef(const LocationDef& locDef);

  RunTypeDef getRunTypeDef() const;
  void setRunTypeDef(const RunTypeDef& runTypeDef);

  // Methods using ID
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // operators
  inline bool operator==(const RunTag& t) const {
    return (m_genTag == t.m_genTag && m_locDef == t.m_locDef && m_runTypeDef == t.m_runTypeDef);
  }

  inline bool operator!=(const RunTag& t) const { return !(*this == t); }

private:
  // User data for this tag
  std::string m_genTag;
  LocationDef m_locDef;
  RunTypeDef m_runTypeDef;

  // Methods from ITag
  int writeDB() noexcept(false);
  void fetchParentIDs(int* locId, int* runTypeID) noexcept(false);

  // Public access methods
  void fetchAllTags(std::vector<RunTag>* fillVec) noexcept(false);
};

#endif
