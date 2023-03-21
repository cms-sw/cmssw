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

#include "tmEventSetup/esScale.hh"

#include <map>
#include <string>
#include <vector>

/**
 *  This class implements data structure for Scale
 */
class L1TUtmScale {
public:
  L1TUtmScale() : name_(), object_(), type_(), minimum_(), maximum_(), step_(), n_bits_(), bins_(), version(0){};

  L1TUtmScale(std::string name,
              int object,
              int type,
              double minimum,
              double maximum,
              double step,
              unsigned int n_bits,
              std::vector<L1TUtmBin> bins,
              unsigned int vers)
      : name_(name),
        object_(object),
        type_(type),
        minimum_(minimum),
        maximum_(maximum),
        step_(step),
        n_bits_(n_bits),
        bins_(bins),
        version(vers){};

  L1TUtmScale(const tmeventsetup::esScale& esSc)
      : name_(esSc.getName()),
        object_(esSc.getObjectType()),
        type_(esSc.getScaleType()),
        minimum_(esSc.getMinimum()),
        maximum_(esSc.getMaximum()),
        step_(esSc.getStep()),
        n_bits_(esSc.getNbits()),
        version(0) {
    bins_.reserve(esSc.getBins().size());
    for (auto it = esSc.getBins().begin(); it != esSc.getBins().end(); ++it)
      bins_.emplace_back(L1TUtmBin(*it));
  };

  virtual ~L1TUtmScale() = default;

  operator tmeventsetup::esScale() const {
    std::vector<tmeventsetup::esBin> bins;
    bins.reserve(getBins().size());
    for (const auto& it : getBins())
      bins.emplace_back(tmeventsetup::esBin(it.hw_index, it.minimum, it.maximum));
    return tmeventsetup::esScale(
        getName(), getObjectType(), getScaleType(), getMinimum(), getMaximum(), getStep(), getNbits(), bins);
  }

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
