//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     8 Nov 2015
 */

#ifndef tmEventSetup_L1TUtmCut_hh
#define tmEventSetup_L1TUtmCut_hh

#include "CondFormats/L1TObjects/interface/L1TUtmCutValue.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

/**
 *  This class implements data structure for Cut
 */
class L1TUtmCut {
public:
  L1TUtmCut() : name_(), object_type_(), cut_type_(), minimum_(), maximum_(), data_(), key_(), version(0){};

  virtual ~L1TUtmCut() = default;

  /** set cut name */
  void setName(const std::string& name) { name_ = name; };

  /** set object type */
  void setObjectType(const int type) { object_type_ = type; };

  /** set cut type */
  void setCutType(const int type) { cut_type_ = type; };

  /** set minimum value of cut range (double) */
  void setMinimumValue(const double value) { minimum_.value = value; };

  /** set minimum value of cut range (HW index) */
  void setMinimumIndex(const unsigned int index) { minimum_.index = index; };

  /** set minimum value of cut range (L1TUtmCutValue struct) */
  void setMinimum(const L1TUtmCutValue& minimum) { minimum_ = minimum; };

  /** set maximum value of cut range (double) */
  void setMaximumValue(const double value) { maximum_.value = value; };

  /** set maximum value of cut range (HW index) */
  void setMaximumIndex(const unsigned int index) { maximum_.index = index; };

  /** set maximum value of cut range (L1TUtmCutValue struct) */
  void setMaximum(const L1TUtmCutValue& maximum) { maximum_ = maximum; };

  /** set precision for cut value calculations */
  void setPrecision(const unsigned int precision) {
    setMaximumIndex(precision);
    setMinimumIndex(precision);
  };  // HACK

  /** get cut name */
  const std::string& getName() const { return name_; };

  /** get target object type : combination of esObjectType and esFunctionType enums */
  const int getObjectType() const { return object_type_; };

  /** get cut type */
  const int getCutType() const { return cut_type_; };

  /** get L1TUtmCutValue struct for minimum value of cut range */
  const L1TUtmCutValue& getMinimum() const { return minimum_; };

  /** get L1TUtmCutValue struct for maximum value of cut range */
  const L1TUtmCutValue& getMaximum() const { return maximum_; };

  /** get L1TUtmCutValue struct for minimum value of cut range */
  const double getMinimumValue() const { return minimum_.value; };

  /** get L1TUtmCutValue struct for maximum value of cut range */
  const double getMaximumValue() const { return maximum_.value; };

  /** get L1TUtmCutValue struct for minimum value of cut range */
  const unsigned int getMinimumIndex() const { return minimum_.index; };

  /** get L1TUtmCutValue struct for maximum value of cut range */
  const unsigned int getMaximumIndex() const { return maximum_.index; };

  /** get data */
  const std::string& getData() const { return data_; };

  /** get key */
  const std::string& getKey() const { return key_; };

  /** get precision */
  const unsigned int getPrecision() const { return getMinimumIndex(); };  // HACK

protected:
  std::string name_;       /**< name of cut */
  int object_type_;        /**< target object type */
  int cut_type_;           /**< type of cut */
  L1TUtmCutValue minimum_; /**< minimum value of cut range */
  L1TUtmCutValue maximum_; /**< maximum value of cut range */
  std::string data_;       /**< data for charge/quality/isolation/charge correlation/impact parameter */
  std::string key_;        /**< key for accessing a scale */
  unsigned int version;
  COND_SERIALIZABLE;
};

#endif  // tmEventSetup_L1TUtmCut_hh
