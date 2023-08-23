#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeAnodeLCTProcessor_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeAnodeLCTProcessor_h

/** \class CSCUpgradeAnodeLCTProcessor
 *
 * This class simulates the functionality of the anode LCT card. It is run by
 * the MotherBoard and returns up to two AnodeLCTs.  It can be run either in a
 * test mode, where it is passed an array of wire times, or in normal mode
 * where it determines the wire times from the wire digis.

 * Updates for high pileup running by Vadim Khotilovich (TAMU), December 2012
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCAnodeLCTProcessor.h"

class CSCUpgradeAnodeLCTProcessor : public CSCAnodeLCTProcessor {
public:
  /** Normal constructor. */
  CSCUpgradeAnodeLCTProcessor(unsigned endcap,
                              unsigned station,
                              unsigned sector,
                              unsigned subsector,
                              unsigned chamber,
                              CSCBaseboard::Parameters& conf);

  /** Default destructor. */
  ~CSCUpgradeAnodeLCTProcessor() override{};

private:
  /* This function looks for LCTs on the previous and next wires.  If one
     exists and it has a better quality and a bx_time up to 4 clocks earlier
     than the present, then the present LCT is cancelled.  The present LCT
     also gets cancelled if it has the same quality as the one on the
     previous wire (this has not been done in 2003 test beam).  The
     cancellation is done separately for collision and accelerator patterns. */
  void ghostCancellationLogicOneWire(const int key_wire, int* ghost_cleared) override;

  /* Quality definition that includes hits in GE1/1 or GE2/1 */
  int getTempALCTQuality(int temp_quality) const override;
};

#endif
