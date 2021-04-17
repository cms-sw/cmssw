//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     12 Mar 2015
 */

#ifndef tmEventSetup_L1TUtmCondition_hh
#define tmEventSetup_L1TUtmCondition_hh

#include "CondFormats/L1TObjects/interface/L1TUtmCut.h"
#include "CondFormats/L1TObjects/interface/L1TUtmObject.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>

/**
 *  This class implements data structure for Condition
 */
class L1TUtmCondition {
public:
  L1TUtmCondition() : name_(), type_(-9999), objects_(), cuts_(), version(0){};

  virtual ~L1TUtmCondition() = default;

  /** set condition name */
  void setName(const std::string& x) { name_ = x; };

  /** set condition type */
  void setType(const int x) { type_ = x; };

  /** get condition name */
  const std::string& getName() const { return name_; };

  /** get condition type */
  const int getType() const { return type_; };

  /** get objects associated with the condition */
  const std::vector<L1TUtmObject>& getObjects() const { return objects_; };

  /** get cuts associated with the condition */
  const std::vector<L1TUtmCut>& getCuts() const { return cuts_; };

protected:
  std::string name_;                  /**< name of condition */
  int type_;                          /**< type of condition */
  std::vector<L1TUtmObject> objects_; /**< list of objects used in condition */
  std::vector<L1TUtmCut> cuts_;       /**< list of cuts applied on condition */
  unsigned int version;
  COND_SERIALIZABLE;
};

#endif  // tmEventSetup_L1TUtmCondition_hh
