#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <fstream>
#include <math.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverMiniLUT.h>

lclphidat CSCSectorReceiverMiniLUT::calcLocalPhiMini(unsigned theadd)
{
  // This method is ripped from CSCSectorReceverLUT.cc with minor changes
  
  lclphidat data;
  
  static int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int pattern = ((theadd >> 8) & 0xf);
  unsigned short int strip   = (theadd & 0xff);
  
  if(strip < 2*CSCConstants::MAX_NUM_STRIPS && pattern < CSCConstants::NUM_CLCT_PATTERNS)
    data.phi_local = static_cast<unsigned>((lcl_phi_param0[pattern] + strip)*lcl_phi_param1);
  else
    throw cms::Exception("CSCSectorReceiverMiniLUT")
      << "+++ Value of strip, " << strip
      << ", exceeds max allowed, " << 2*CSCConstants::MAX_NUM_STRIPS-1
      << " +++\n";
  
  if(data.phi_local >= maxPhiL)
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of phi_local, " << data.phi_local
    << ", exceeds max allowed, " << CSCConstants::NUM_CLCT_PATTERNS-1 << " +++\n";
  
  data.phi_bend_local = 0;
    
  return data;
}

global_eta_data CSCSectorReceiverMiniLUT::calcGlobalEtaMEMini(unsigned short endcap, unsigned short sector, unsigned short station, unsigned short subsector, unsigned theadd)
{
  gbletadat data(0);
  
  unsigned short int cscid  = ((theadd >> 15) & 0xf);
  unsigned short int lclPhi = ((theadd >> 6)  & 0x3);
  unsigned short int WG     = ((theadd >> 8)  & 0x7f);
  
  int eta_temp;

  if((cscid > 0) && (cscid <= 9) && (WG < CSCConstants::MAX_NUM_WIRES))
    eta_temp =  (gbl_eta_params[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][0] + 
                 gbl_eta_params[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][1] *
                 log(gbl_eta_params[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][2] + WG));
  else
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of cscid, " << cscid
    << ", is out of bounds, [1, 9] -- or --"
    << " Value of wire group, " << WG 
    << ", exceeds max allowed, " << CSCConstants::MAX_NUM_WIRES << " +++\n";
  
  // protect from negative numbers.  If the value of eta_temp is <0, set global eta to the minimum value
  if((eta_temp >= gbl_eta_bounds[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][0]) &&
     (eta_temp <= gbl_eta_bounds[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][1]))
    data.global_eta = eta_temp;
  else if(eta_temp < gbl_eta_bounds[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][0])
    data.global_eta = gbl_eta_bounds[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][0];
  else
    data.global_eta = gbl_eta_bounds[endcap-1][sector-1][station-1][subsector-1][lclPhi][cscid-1][1];
  
  data.global_bend = 0;
  return data;
}

global_phi_data CSCSectorReceiverMiniLUT::calcGlobalPhiMEMini(unsigned short endcap, unsigned short sector, unsigned short station, unsigned short subsector, unsigned theadd)
{
  gblphidat data(0);
  
  unsigned short int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int maxPhiG = 1<<CSCBitWidths::kGlobalPhiDataBitWidth;
  unsigned short int cscid  = ((theadd >> 15)&0xf);
  unsigned short int lclPhi = (theadd & 0x3ff);

  if(station == 1 && ((cscid <= 3) || (cscid >= 7))) maxPhiL = maxPhiL*(64./80); // currently a hack that is in place to handle the different number of strips in ME1/1 and ME1/3
  
  if((cscid > 0) && (cscid <= 9) && (lclPhi < maxPhiL))
    data.global_phi = (gbl_phi_me_params[endcap-1][sector-1][station-1][subsector-1][cscid-1][0] + 
                       gbl_phi_me_params[endcap-1][sector-1][station-1][subsector-1][cscid-1][1]*lclPhi);
  else if((cscid > 0) && (cscid <= 9) && (lclPhi >= maxPhiL))
    data.global_phi = (gbl_phi_me_params[endcap-1][sector-1][station-1][subsector-1][cscid-1][0] + 
                       gbl_phi_me_params[endcap-1][sector-1][station-1][subsector-1][cscid-1][1]*(maxPhiL-1));
  else
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of cscid, " << cscid
    << ", is out of bounds, [1, 9] +++\n";
  
  if(data.global_phi >= maxPhiG)
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of global_phi, " << data.global_phi
    << ", exceeds max allowed, " << maxPhiG-1 << " +++\n";
  
  return data;
}

global_phi_data CSCSectorReceiverMiniLUT::calcGlobalPhiMBMini(unsigned short endcap, unsigned short sector, unsigned short subsector, unsigned theadd)
{
  gblphidat data(0);
  
  unsigned short int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  unsigned short int maxPhiG = 1<<CSCBitWidths::kGlobalPhiDataBitWidth;
  unsigned short int cscid  = ((theadd >> 15)&0xf);
  unsigned short int lclPhi = (theadd & 0x3ff);
  
  if((cscid <= 3) || (cscid >= 7)) maxPhiL = maxPhiL*(64./80); // currently a hack that is in place to handle the different number of strips in ME1/1 and ME1/3
  
  if((cscid > 0) && (cscid <= 9) && (lclPhi < maxPhiL))
    data.global_phi = gbl_phi_mb_params[endcap-1][sector-1][subsector-1][cscid-1][0] + gbl_phi_mb_params[endcap-1][sector-1][subsector-1][cscid-1][1]*lclPhi;
  else if((cscid > 0) && (cscid <= 9) && (lclPhi >= maxPhiL))
    data.global_phi = gbl_phi_mb_params[endcap-1][sector-1][subsector-1][cscid-1][0] + gbl_phi_mb_params[endcap-1][sector-1][subsector-1][cscid-1][1]*(maxPhiL-1);
  else
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of cscid, " << cscid
    << ", is out of bounds, [1, 9] -- or --"
    << " Value of lclPhi, " << lclPhi 
    << ", exceeds max allowed, " << maxPhiL << " +++\n";
  
  if(data.global_phi >= maxPhiG)
    throw cms::Exception("CSCSectorReceiverMiniLUT")
    << "+++ Value of global_phi, " << data.global_phi
    << ", exceeds max allowed, " << maxPhiG-1 << " +++\n";
  
  return data;
}
