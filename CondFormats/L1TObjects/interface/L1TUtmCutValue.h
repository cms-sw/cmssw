//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     12 Mar 2015
 */

/** @todo nope */

#ifndef tmEventSetup_L1TUtmCutValue_hh
#define tmEventSetup_L1TUtmCutValue_hh

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
 *  This class implements data structure for CutValue
 */
struct L1TUtmCutValue
{
  // ctor
  L1TUtmCutValue()
    : value(std::numeric_limits<double>::max()),
      index(std::numeric_limits<unsigned int>::max()),
      version(0) { };

  double value;               /**< cut value */
  unsigned int index;         /**< HW index for the cut value */
  unsigned int version;
  COND_SERIALIZABLE;
};

#endif // tmEventSetup_L1TUtmCutValue_hh
/* eof */
