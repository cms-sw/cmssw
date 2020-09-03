#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <DataFormats/L1TMuon/interface/CSCConstants.h>

#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <fstream>
#include <cmath>
#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverMiniLUT.h>

lclphidat CSCSectorReceiverMiniLUT::calcLocalPhiMini(unsigned theadd, const bool gangedME1a) {
  // This method is ripped from CSCSectorReceverLUT.cc with minor changes

  lclphidat data;

  constexpr int maxPhiL = 1 << CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int pattern = ((theadd >> 8) & 0xf);
  unsigned short int strip = (theadd & 0xff);

  if (strip < 2 * (CSCConstants::MAX_NUM_STRIPS * 7 / 5) &&
      pattern <
          CSCConstants::
              NUM_CLCT_PATTERNS) {  // MDG, DA and RW, for ME1 we have 7CFEBs and not just 5, so the num_strips can go up to 16 * 7 but only for ME1
    data.phi_local = gangedME1a ? static_cast<unsigned>((lcl_phi_param0[pattern] + strip) * lcl_phi_param1)
                                : static_cast<unsigned>((lcl_phi_param0[pattern] + strip) * 0.625 * lcl_phi_param1);
    //DA and MDG, rescale range of local phi so ME1/1b fits in 0-511
  } else
    edm::LogWarning("CSCSectorReceiverMiniLUT") << "+++ Value of strip, " << strip << ", exceeds max allowed, "
                                                << 2 * CSCConstants::MAX_NUM_STRIPS - 1 << " +++\n";

  if (data.phi_local >= maxPhiL)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of phi_local, " << data.phi_local << ", exceeds max allowed, "
        << CSCConstants::NUM_CLCT_PATTERNS - 1 << " +++\n";

  //  data.phi_bend_local = 0;
  // Just pass through all bits of pattern as bend angle (so 2 MSB unfilled)
  data.phi_bend_local = pattern & 0x3F;

  return data;
}

global_eta_data CSCSectorReceiverMiniLUT::calcGlobalEtaMEMini(unsigned short endcap,
                                                              unsigned short sector,
                                                              unsigned short station,
                                                              unsigned short subsector,
                                                              unsigned theadd,
                                                              const bool gangedME1a) {
  if (endcap < 1 || endcap > 2)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of endcap, " << endcap << ", is out of bounds, [1, 2] +++\n";
  if (sector < 1 || sector > 6)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of sector, " << sector << ", is out of bounds, [1, 6] +++\n";
  if (station < 1 || station > 4)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of station, " << station << ", is out of bounds, [1, 4] +++\n";

  gbletadat data(0);

  unsigned short int tcscid = ((theadd >> 15) & 0xf);
  unsigned short int lclPhi = ((theadd >> 6) & 0x3);
  unsigned short int WG = ((theadd >> 8) & 0x7f);
  unsigned short int bend = ((theadd)&0x3f);

  int eta_temp = 999, eta_min = 999, eta_max = 999;

  if ((tcscid > 0) && (tcscid <= 12) && (WG < CSCConstants::MAX_NUM_WIRES)) {
    unsigned short int cscid = (tcscid > 9) ? tcscid - 9 : tcscid;
    if (station == 1) {
      unsigned short int lclPhip = 0;
      if (lclPhi == 1 || lclPhi == 3)
        lclPhip = 2;
      // use only eta correction for first and last third of ME1/1 chamber since local phi scaling changed
      if (gangedME1a) {
        eta_temp =
            (gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhi][cscid - 1][0] +
             gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhi][cscid - 1][1] *
                 log(gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhi][cscid - 1][2] + WG));
        eta_min = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhi][cscid - 1][0];
        eta_max = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhi][cscid - 1][1];
      } else {  // DA and MDG, if unganged replace "lclPhi" index with "lclPhip"
        eta_temp =
            (gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhip][cscid - 1][0] +
             gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhip][cscid - 1][1] *
                 log(gbl_eta_params[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhip][cscid - 1][2] + WG));
        eta_min = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhip][cscid - 1][0];
        eta_max = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][subsector - 1][lclPhip][cscid - 1][1];
      }

      // add offset to ME+11a, subtract for ME-11a (wire tilt and strip direction)1
      // only is ganged
      if (gangedME1a && (tcscid < 4) && (lclPhi == 3)) {
        if (endcap == 1)
          eta_temp += 3;
        else
          eta_temp -= 3;
      }
    } else {
      eta_temp = (gbl_eta_params[endcap - 1][sector - 1][station - 1][0][lclPhi][cscid - 1][0] +
                  gbl_eta_params[endcap - 1][sector - 1][station - 1][0][lclPhi][cscid - 1][1] *
                      log(gbl_eta_params[endcap - 1][sector - 1][station - 1][0][lclPhi][cscid - 1][2] + WG));
      eta_min = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][0][lclPhi][cscid - 1][0];
      eta_max = gbl_eta_bounds[endcap - 1][sector - 1][station - 1][0][lclPhi][cscid - 1][1];
    }
  } else {
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of cscid, " << tcscid << ", is out of bounds, [1, 9] -- or --"
        << " Value of wire group, " << WG << ", exceeds max allowed, " << CSCConstants::MAX_NUM_WIRES << " +++\n";
  }

  // protect from negative numbers.  If the value of eta_temp is <0, set global eta to the minimum value
  if ((eta_temp >= eta_min) && (eta_temp <= eta_max))
    data.global_eta = eta_temp;
  else if (eta_temp < eta_min)
    data.global_eta = eta_min;
  else
    data.global_eta = eta_max;

  //  data.global_bend = 0;
  // Just pass through lowest 5 bits of local bend (drop 1 MSB)
  data.global_bend = bend & 0x1F;

  return data;
}

global_phi_data CSCSectorReceiverMiniLUT::calcGlobalPhiMEMini(unsigned short endcap,
                                                              unsigned short sector,
                                                              unsigned short station,
                                                              unsigned short subsector,
                                                              unsigned theadd,
                                                              const bool gangedME1a) {
  if (endcap < 1 || endcap > 2)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of endcap, " << endcap << ", is out of bounds, [1, 2] +++\n";
  if (sector < 1 || sector > 6)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of sector, " << sector << ", is out of bounds, [1, 6] +++\n";
  if (station < 1 || station > 4)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of station, " << station << ", is out of bounds, [1, 4] +++\n";

  gblphidat data(0);

  unsigned short int maxPhiL = 1 << CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int maxPhiG = 1 << CSCBitWidths::kGlobalPhiDataBitWidth;
  unsigned short int cscid = ((theadd >> 15) & 0xf);
  unsigned short int lclPhi = (theadd & 0x3ff);
  if (!gangedME1a)
    lclPhi = lclPhi / 0.625;  // DA and MDG recover old scaling of local phi

  // 12/11/09
  // GP et DA: how to identify the strip number and isolate and shift the localPhi value
  const double binPhiL = static_cast<double>(maxPhiL) / (2 * CSCConstants::MAX_NUM_STRIPS);

  int strip = static_cast<int>(lclPhi / binPhiL);
  if (station == 1 && (cscid <= 3) &&
      (strip >= 127 && strip < 224)) {  // 160 --> 224, change range for ME1/1a acceptance, DA and MDG
    // in this case need to redefine lclPhi in order to
    // place local phi in the middle of the 5th CFEB
    // and not on the first third of the CFEB as default

    gangedME1a
        ? lclPhi = (strip - 127 + 31) * (4 * binPhiL / 3)
        : lclPhi =
              (strip - 127) *
              (4 * binPhiL /
               3);  //DA and MDG remove offset to center of ME1/1a (no ganging), and reset ME1/1a strip number to start from 0, and scale 48 strips to match ME1/1b 64 strips
  }
  // end GP et DA

  if (station == 1 && ((cscid <= 3) || (cscid >= 7))) {
    //if ( (strip >= 127 && strip < 160) || (cscid >= 10) ) // VK: the || (cscid >= 10) for unganged ME1a
    //  maxPhiL = maxPhiL*(48./80); // GP et DA: currently a hack that is in place to handle the different number of strips in ME1/1a and ME1/3
    //else
    maxPhiL =
        maxPhiL *
        (64. / 80);  // currently a hack that is in place to handle the different number of strips in ME1/1 and ME1/3
  }

  // VK: The the unganged ME1a hack
  if (station == 1 && (cscid >= 10)) {
    lclPhi = strip * (4 * binPhiL / 3);
    cscid = cscid - 9;  // back to normal 1-9 range
  }
  // end VK

  if ((cscid > 0) && (cscid <= 9)) {
    if ((station == 1) && (lclPhi < maxPhiL))
      data.global_phi = (gbl_phi_me_params[endcap - 1][sector - 1][station - 1][subsector - 1][cscid - 1][0] +
                         gbl_phi_me_params[endcap - 1][sector - 1][station - 1][subsector - 1][cscid - 1][1] * lclPhi);
    else if ((station == 1) && (lclPhi >= maxPhiL))
      data.global_phi =
          (gbl_phi_me_params[endcap - 1][sector - 1][station - 1][subsector - 1][cscid - 1][0] +
           gbl_phi_me_params[endcap - 1][sector - 1][station - 1][subsector - 1][cscid - 1][1] * (maxPhiL - 1));
    else
      data.global_phi = (gbl_phi_me_params[endcap - 1][sector - 1][station - 1][0][cscid - 1][0] +
                         gbl_phi_me_params[endcap - 1][sector - 1][station - 1][0][cscid - 1][1] * lclPhi);
  } else
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of cscid, " << cscid << ", is out of bounds, [1, 9] +++\n";

  if (data.global_phi >= maxPhiG)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of global_phi, " << data.global_phi << ", exceeds max allowed, " << maxPhiG - 1 << " +++\n";

  return data;
}

global_phi_data CSCSectorReceiverMiniLUT::calcGlobalPhiMBMini(
    unsigned short endcap, unsigned short sector, unsigned short subsector, unsigned theadd, const bool gangedME1a) {
  if (endcap < 1 || endcap > 2)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of endcap, " << endcap << ", is out of bounds, [1, 2] +++\n";
  if (sector < 1 || sector > 6)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of sector, " << sector << ", is out of bounds, [1, 6] +++\n";

  gblphidat data(0);

  unsigned short int maxPhiL = 1 << CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int maxPhiG = 1 << CSCBitWidths::kGlobalPhiDataBitWidth;
  unsigned short int cscid = ((theadd >> 15) & 0xf);
  unsigned short int lclPhi = (theadd & 0x3ff);
  if (!gangedME1a)
    lclPhi = lclPhi / 0.625;  // DA and MDG, recover old scaling of local phi

  if ((cscid <= 3) || (cscid >= 7))
    maxPhiL =
        maxPhiL *
        (64. / 80);  // currently a hack that is in place to handle the different number of strips in ME1/1 and ME1/3

  if ((cscid > 0) && (cscid <= 9)) {
    if (lclPhi < maxPhiL)
      data.global_phi = (gbl_phi_mb_params[endcap - 1][sector - 1][subsector - 1][cscid - 1][0] +
                         gbl_phi_mb_params[endcap - 1][sector - 1][subsector - 1][cscid - 1][1] * lclPhi);
    else
      data.global_phi = (gbl_phi_mb_params[endcap - 1][sector - 1][subsector - 1][cscid - 1][0] +
                         gbl_phi_mb_params[endcap - 1][sector - 1][subsector - 1][cscid - 1][1] * (maxPhiL - 1));
  } else
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of cscid, " << cscid << ", is out of bounds, [1, 9] +++\n";

  if (data.global_phi >= maxPhiG)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of global_phi, " << data.global_phi << ", exceeds max allowed, " << maxPhiG - 1 << " +++\n";

  if (data.global_phi >= maxPhiG)
    edm::LogWarning("CSCSectorReceiverMiniLUT")
        << "+++ Value of global_phi, " << data.global_phi << ", exceeds max allowed, " << maxPhiG - 1 << " +++\n";

  return data;
}
