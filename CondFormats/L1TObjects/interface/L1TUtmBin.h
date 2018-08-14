//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     12 Mar 2015
 */

/** @todo nope */

#ifndef tmEventSetup_L1TUtmBin_hh
#define tmEventSetup_L1TUtmBin_hh

/*====================================================================*
 * declarations
 *====================================================================*/
/*-----------------------------------------------------------------*
 * headers
 *-----------------------------------------------------------------*/
#include <limits>
#include "CondFormats/Serialization/interface/Serializable.h"


/*-----------------------------------------------------------------*
 * constants
 *-----------------------------------------------------------------*/
/* nope */



/**
 *  This class implements data structure for Bin
 */
class L1TUtmBin
{
  public:
    // ctor
    L1TUtmBin()
      : hw_index(std::numeric_limits<unsigned int>::max()),
        minimum(std::numeric_limits<double>::min()),
        maximum(std::numeric_limits<double>::max()),
        version(0) { };

    L1TUtmBin(const unsigned int id, 
          const double min,
          const double max)
      : hw_index(id), minimum(min), maximum(max), version(0) { };

    // dtor
    virtual ~L1TUtmBin() { };

    unsigned int hw_index;   /**< HW index of bin */
    double minimum;          /**< minimum value of bin */
    double maximum;          /**< maximum value of bin */
    unsigned int version;
  COND_SERIALIZABLE;
};

#endif // tmEventSetup_L1TUtmBin_hh
/* eof */
