//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     9 Nov 2015
 */

#ifndef tmEventSetup_L1TUtmScale_hh
#define tmEventSetup_L1TUtmScale_hh

#include "CondFormats/L1TObjects/interface/L1TUtmBin.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>
#include <vector>

/**
 *  This class implements data structure for Scale
 */
class L1TUtmScale {
public:
  L1TUtmScale() : name_(), object_(), type_(), minimum_(), maximum_(), step_(), n_bits_(), bins_(), version(0){};

  virtual ~L1TUtmScale() = default;

  /** get scale name */
  const std::string& getName() const { return name_; };

  /** get target object type */
  int getObjectType() const { return object_; };

  /** get scale type */
  int getScaleType() const { return type_; };

  /** get minimum value of the scale */
  double getMinimum() const { return minimum_; };

  /** get maximum value of the scale */
  double getMaximum() const { return maximum_; };

  /** get step size of linear scale */
  double getStep() const { return step_; };

  /** get number of bits for the scale */
  unsigned int getNbits() const { return n_bits_; };

  /** get bins for the scale */
  const std::vector<L1TUtmBin>& getBins() const { return bins_; };

protected:
  std::string name_;            /**< name of scale */
  int object_;                  /**< type of object */
  int type_;                    /**< type of scale */
  double minimum_;              /**< minimum value of scale */
  double maximum_;              /**< maximum value of scale */
  double step_;                 /**< step size of linear scale */
  unsigned int n_bits_;         /**< number of bits for scale */
  std::vector<L1TUtmBin> bins_; /**< array of L1TUtmBin */
  unsigned int version;
  COND_SERIALIZABLE;
};

#endif  // tmEventSetup_L1TUtmScale_hh
