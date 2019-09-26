#ifndef L1Trigger_CSCTriggerPrimitives_LCTContainer_h
#define L1Trigger_CSCTriggerPrimitives_LCTContainer_h

/** \class LCTContainer
 *
 * This class is a helper class that contains LCTs per BX
 *
 * Author: Nick McColl
 *
 */

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <vector>
#include <algorithm>

/** for the case when more than 2 LCTs/BX are allowed;
    maximum match window = 15 */
class LCTContainer {
public:
  // constructor
  LCTContainer(unsigned int trig_window_size);

  // access the LCT in a particular ALCT BX, a particular CLCT matched BX
  // and particular LCT number
  CSCCorrelatedLCTDigi& operator()(int bx, int match_bx, int lct);

  // get the matching LCTs for a certain ALCT BX
  void getTimeMatched(const int bx, std::vector<CSCCorrelatedLCTDigi>&) const;

  // get all LCTs in the 16 BX readout window
  void getMatched(std::vector<CSCCorrelatedLCTDigi>&) const;

  // clear the array with stubs
  void clear();

  // array with stored LCTs
  // 1st index: depth of pipeline that stores the ALCT and CLCT
  // 2nd index: BX number of the ALCT-CLCT match in the matching window
  // 3rd index: LCT number in the time bin
  CSCCorrelatedLCTDigi data[CSCConstants::MAX_LCT_TBINS][CSCConstants::MAX_MATCH_WINDOW_SIZE]
                           [CSCConstants::MAX_LCTS_PER_CSC];

  // matching trigger window
  const unsigned int match_trig_window_size_;
};

#endif
