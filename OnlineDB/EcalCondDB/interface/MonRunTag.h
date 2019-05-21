#ifndef MONRUNTAG_H
#define MONRUNTAG_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/MonVersionDef.h"

/**
 *   Tag for Monitoring Sub-Run information
 */
class MonRunTag : public ITag {
public:
  friend class MonRunIOV;  // needs permission to write
  friend class EcalCondDBInterface;

  MonRunTag();
  ~MonRunTag() override;

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  MonVersionDef getMonVersionDef() const;
  void setMonVersionDef(const MonVersionDef& ver);

  // Methods using ID
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators
  inline bool operator==(const MonRunTag& t) const {
    return (m_genTag == t.m_genTag && m_monVersionDef == t.m_monVersionDef);
  }

  inline bool operator!=(const MonRunTag& t) const { return !(*this == t); }

private:
  // User data for this tag
  std::string m_genTag;
  MonVersionDef m_monVersionDef;

  // Methods from ITag
  int writeDB() noexcept(false);

  // Access methods
  void fetchAllTags(std::vector<MonRunTag>* fillVec) noexcept(false);

  void fetchParentIDs(int* verID) noexcept(false);
};

#endif
