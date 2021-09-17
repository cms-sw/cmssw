#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeAnodeLCTProcessor.h"

CSCUpgradeAnodeLCTProcessor::CSCUpgradeAnodeLCTProcessor(unsigned endcap,
                                                         unsigned station,
                                                         unsigned sector,
                                                         unsigned subsector,
                                                         unsigned chamber,
                                                         const edm::ParameterSet& conf)
    : CSCAnodeLCTProcessor(endcap, station, sector, subsector, chamber, conf) {
  if (!runPhase2_)
    edm::LogError("CSCUpgradeAnodeLCTProcessor|ConfigError")
        << "+++ Upgrade CSCUpgradeAnodeLCTProcessor constructed while runPhase2_ is not set! +++\n";

  if (!enableAlctPhase2_)
    edm::LogError("CSCUpgradeAnodeLCTProcessor|ConfigError")
        << "+++ Upgrade CSCUpgradeAnodeLCTProcessor constructed while enableAlctPhase2_ is not set! +++\n";
}

void CSCUpgradeAnodeLCTProcessor::ghostCancellationLogicOneWire(const int key_wire, int* ghost_cleared) {
  for (int i_pattern = 0; i_pattern < 2; i_pattern++) {
    ghost_cleared[i_pattern] = 0;
    if (key_wire == 0)
      continue;

    // Non-empty wire group.
    int qual_this = quality[key_wire][i_pattern];
    if (qual_this > 0) {
      if (runPhase2_ and runME21ILT_ and isME21_)
        qual_this = (qual_this & 0x03);
      // Previous wire.
      int dt = -1;
      for (auto& p : lct_list) {
        if (not(p.isValid() and p.getKeyWG() == key_wire - 1 and 1 - p.getAccelerator() == i_pattern))
          continue;

        bool ghost_cleared_prev = false;
        int qual_prev = p.getQuality();
        int first_bx_prev = p.getBX();
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor")
              << "ghost concellation logic " << ((i_pattern == 0) ? "Accelerator" : "Collision") << " key_wire "
              << key_wire << " quality " << qual_this << " bx " << first_bx[key_wire] << " previous key_wire "
              << key_wire - 1 << " quality " << qual_prev << " bx " << first_bx[key_wire - 1];

        //int dt = first_bx[key_wire] - first_bx[key_wire-1];
        if (use_corrected_bx)
          dt = first_bx_corrected[key_wire] - first_bx_prev;
        else
          dt = first_bx[key_wire] - first_bx_prev;
        // hack to run the Phase-II ME2/1, ME3/1 and ME4/1 ILT
        if (runPhase2_ and runME21ILT_ and isME21_)
          qual_prev = (qual_prev & 0x03);

        // Cancel this wire
        //   1) If the candidate at the previous wire is at the same bx
        //      clock and has better quality (or equal? quality - this has
        //      been implemented only in 2004).
        //   2) If the candidate at the previous wire is up to 4 clocks
        //      earlier, regardless of quality.
        if (dt == 0) {
          if (qual_prev > qual_this)
            ghost_cleared[i_pattern] = 1;
        } else if (dt > 0 && dt <= ghost_cancellation_bx_depth) {
          if ((!ghost_cancellation_side_quality) || (qual_prev > qual_this))
            ghost_cleared[i_pattern] = 1;
        } else if (dt < 0 && dt * (-1) <= ghost_cancellation_bx_depth) {
          if ((!ghost_cancellation_side_quality) || (qual_prev < qual_this))
            ghost_cleared_prev = true;
        }

        if (ghost_cleared[i_pattern] == 1) {
          if (infoV > 1)
            LogTrace("CSCUpgradeAnodeLCTProcessor")
                << ((i_pattern == 0) ? "Accelerator" : "Collision") << " pattern ghost cancelled on key_wire "
                << key_wire << " q=" << qual_this << "  by wire " << key_wire - 1 << " q=" << qual_prev
                << "  dt=" << dt;
          //cancellation for key_wire is done when ALCT is created and pushed to lct_list
        }
        if (ghost_cleared_prev) {
          if (infoV > 1)
            LogTrace("CSCAnodeLCTProcessor")
                << ((i_pattern == 0) ? "Accelerator" : "Collision") << " pattern ghost cancelled on key_wire "
                << key_wire - 1 << " q=" << qual_prev << "  by wire " << key_wire << " q=" << qual_this;
          p.setValid(0);  //clean prev ALCT
        }
      }
    }
  }
}

int CSCUpgradeAnodeLCTProcessor::getTempALCTQuality(int temp_quality) const {
  // Quality definition changed on 22 June 2007: it no longer depends
  // on pattern_thresh.
  int Q;
  // hack to run the Phase-II GE2/1-ME2/1 with 3-layer ALCTs
  if (temp_quality == 3 and runPhase2_ and runME21ILT_ and isME21_)
    Q = 1;
  else if (temp_quality > 3)
    Q = temp_quality - 3;
  else
    Q = 0;  // quality code 0 is valid!

  return Q;
}
