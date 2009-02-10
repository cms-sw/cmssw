#ifndef HCALIGNORECELLSALGO_H
#define HCALIGNORECELLSALGO_H 1


#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

/** \class HcalIgnoreCellsAlgo
    This class keeps track of cells that have been marked as bad via 
    their status words in HcalChannelStatus.  The ignorebadcells
    method returns a boolean 'true' if the cell's channel status is bad 
    (dead, hot, cell off, etc.), and a boolean 'false' otherwise.

    $Date: 2008/11/28 10:42:00 %
    $Revision: 1.1 %
    \author J. Temple -- Univ. of Maryland
*/


class HcalIgnoreCellsAlgo{
 public:
  // Constructors
  HcalIgnoreCellsAlgo(); // default constructor; sets mask to 0
  HcalIgnoreCellsAlgo(int mask);
  // Destructor
  ~HcalIgnoreCellsAlgo();
  
  // Method for determining whether cells should be ignored
  bool ignoreBadCells(DetId& id,  HcalChannelQuality* myqual);

  int statusMask_;
};

#endif
  
