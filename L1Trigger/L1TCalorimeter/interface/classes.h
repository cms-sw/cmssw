/*
 * ========================================================================
 *
 *    Description:  Define dataformats for L1RecoMatch, L1 Global object.
 *
 * ========================================================================
 */

#include "L1Trigger/L1TCalorimeter/interface/L1RecoMatch.h"
#include "L1Trigger/L1TCalorimeter/interface/L1GObject.h"

namespace {

  L1RecoMatch dummyMatch;
  L1GObject dummyL1G;
  std::vector<L1GObject> dummyL1GCollection;
  edm::Wrapper<std::vector<L1GObject> > dummyL1GWrapper;

}
