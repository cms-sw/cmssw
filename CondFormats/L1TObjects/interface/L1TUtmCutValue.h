//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     12 Mar 2015
 */

#ifndef tmEventSetup_L1TUtmCutValue_hh
#define tmEventSetup_L1TUtmCutValue_hh

#include <limits>
#include "CondFormats/Serialization/interface/Serializable.h"

/**
 *  This class implements data structure for CutValue
 */
struct L1TUtmCutValue {
  L1TUtmCutValue()
      : value(std::numeric_limits<double>::max()), index(std::numeric_limits<unsigned int>::max()), version(0){};

  virtual ~L1TUtmCutValue() = default;

  double value;       /**< cut value */
  unsigned int index; /**< HW index for the cut value */
  unsigned int version;
  COND_SERIALIZABLE;
};

#endif  // tmEventSetup_L1TUtmCutValue_hh
