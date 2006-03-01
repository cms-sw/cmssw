#ifndef LMFRUNTAG_H
#define LMFRUNTAG_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"


/**
 *   Tag for Monitoring Sub-Run information
 */
class LMFRunTag : public ITag {
 public:
  friend class LMFRunIOV;  // needs permission to write
  friend class EcalCondDBInterface;

  LMFRunTag();
  ~LMFRunTag();

  // Methods for user data
  std::string getGeneralTag() const;
  void setGeneralTag(std::string tag);

  // Methods using ID
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);


 private:
  // User data for this tag
  std::string m_genTag;

  // Methods from ITag
  int writeDB() throw(std::runtime_error);

  // Access methods
  void fetchAllTags( std::vector<LMFRunTag>* fillVec) throw(std::runtime_error);


};

#endif
