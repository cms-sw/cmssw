#ifndef DCUTAG_H
#define DCUTAG_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/LocationDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
/**
 *   Tag for Run information
 */
class DCUTag : public ITag {
public:
  friend class DCUIOV;  // needs permission to write
  friend class EcalCondDBInterface;

  DCUTag();
  ~DCUTag() override;

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  LocationDef getLocationDef() const;
  void setLocationDef(const LocationDef& locDef);

  // Methods using ID
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators
  inline bool operator==(const DCUTag& t) const { return (m_genTag == t.m_genTag && m_locDef == t.m_locDef); }

  inline bool operator!=(const DCUTag& t) const { return !(*this == t); }

private:
  // User data for this tag
  std::string m_genTag;
  LocationDef m_locDef;

  // Methods from ITag
  int writeDB() noexcept(false);
  void fetchParentIDs(int* locId) noexcept(false);

  // Public access methods
  void fetchAllTags(std::vector<DCUTag>* fillVec) noexcept(false);
};

#endif
