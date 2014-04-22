#ifndef CSCTriggerPrimitives_CSCGEMRPCTriggerGeometryHelper_h
#define CSCTriggerPrimitives_CSCGEMRPCTriggerGeometryHelper_h

/** \class CSCGEMRPCTriggerGeometryHelper
 *
 * Collection of LUTs and helper functions for the 
 * GEM-CSC and CSC-RPC local integrated triggers
 * for the Phase-II Muon upgrade
 *
 * \author Sven Dildick April 2014
 *
 */

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/Math/interface/deltaPhi.h>
#include <DataFormats/Math/interface/normalizedPhi.h>

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;

class CSCGEMRPCTriggerGeometryHelper
{
 public:
  CSCGEMRPCTriggerGeometryHelper();
  ~CSCGEMRPCTriggerGeometryHelper();

  void setup();

  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }
  void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }

  int cscHalfStripToGEMPad(int st, int hs, bool isEven);
  int gemPadToCSCHalfStrip(int st, int pad, bool isEven);
  
  int cscHalfStripToRPCStrip(int st, int hs, bool isEven);
  int rpcStripToCSCHalfStrip(int st, int pad, bool isEven);

 private:
  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;
  const RPCGeometry* rpc_g;

  // ME11 wire groups are slanted --> min & max 
  static const double me11_wg_eta_odd[48][3];
  static const double me11_wg_eta_even[48][3];

  // ME21, ME31 and ME41 wire groups run horizontal
  static const double me21_wg_eta_odd[112][2];
  static const double me21_wg_eta_even[112][2];

  static const double me31_wg_eta_odd[96][2];
  static const double me31_wg_eta_even[96][2];

  static const double me41_wg_eta_odd[96][2];
  static const double me41_wg_eta_even[96][2];

  // map GEM or RPC rolls onto wiregroups
  static const double me11_gem_roll_eta_odd[10][3];
  static const double me11_gem_roll_eta_even[10][3];

  static const double me21_short_gem_roll_eta_odd[10][3];
  static const double me21_short_gem_roll_eta_even[10][3];

  static const double me21_long_gem_roll_eta_odd[10][3];
  static const double me21_long_gem_roll_eta_even[10][3];

  static const double me31_rpc_roll_eta_odd[10][3];
  static const double me31_rpc_roll_eta_even[10][3];

  static const double me41_rpc_roll_eta_odd[10][3];
  static const double me41_rpc_roll_eta_even[10][3];

  // map wiregroups onto rolls (GEM or RPC)
  static const int me11_wg_gem_roll_odd[10][3];
  static const int me11_wg_gem_roll_even[10][3];

  static const int me21_short_wg_gem_roll_odd[10][3];
  static const int me21_short_wg_gem_roll_even[10][3];

  static const int me21_long_wg_gem_roll_odd[10][3];
  static const int me21_long_wg_gem_roll_even[10][3];

  static const int me31_wg_gem_roll_odd[10][3];
  static const int me31_wg_gem_roll_even[10][3];

  static const int me41_wg_gem_roll_odd[10][3];
  static const int me41_wg_gem_roll_even[10][3];

  // map GEM pads or RPC strips to CSC half-strips
  static const int me1a_gem_pad_hs[192][2];
  static const int me1b_gem_pad_hs[192][2];

  static const int me21_gem_pad_hs[192][2];

  static const int me31_rpc_strip_hs[192][2];

  static const int me41_rpc_strip_hs[192][2];

  // map CSC half-strips onto GEM pads or RPC strips
  static const int me1a_hs_gem_pad[192][2];

  static const int me1b_hs_gem_pad[192][2];

  static const int me21_hs_gem_pad[192][2];

  static const int me31_hs_rpc_strip[192][2];
    
  static const int me41_hs_rpc_strip[192][2];
};

#endif

/*
  // loop on all wiregroups to create a LUT <WG,rollMin,rollMax>
  int numberOfWG(cscChamber->layer(1)->geometry()->numberOfWireGroups());
  std::cout <<"detId " << cscChamber->id() << std::endl;
  for (int i = 0; i< numberOfWG; ++i){
    // find low-eta of WG
    auto length(cscChamber->layer(1)->geometry()->lengthOfWireGroup(i));
//     auto gp(cscChamber->layer(1)->centerOfWireGroup(i));
    auto lpc(cscChamber->layer(1)->geometry()->localCenterOfWireGroup(i));
    auto wireEnds(cscChamber->layer(1)->geometry()->wireTopology()->wireEnds(i));
    auto gpMin(cscChamber->layer(1)->toGlobal(wireEnds.first));
    auto gpMax(cscChamber->layer(1)->toGlobal(wireEnds.second));
    auto etaMin(gpMin.eta());
    auto etaMax(gpMax.eta());
    if (etaMax < etaMin)
      std::swap(etaMin,etaMax);
    //print the eta min and eta max
    //    std::cout << i << " " << etaMin << " " << etaMax << std::endl;
    auto x1(lpc.x() + cos(cscChamber->layer(1)->geometry()->wireAngle())*length/2.);
    auto x2(lpc.x() - cos(cscChamber->layer(1)->geometry()->wireAngle())*length/2.);
    auto z(lpc.z());
    auto y1(cscChamber->layer(1)->geometry()->yOfWireGroup(i,x1));
    auto y2(cscChamber->layer(1)->geometry()->yOfWireGroup(i,x2));
    auto lp1(LocalPoint(x1,y1,z));
    auto lp2(LocalPoint(x2,y2,z));
    auto gp1(cscChamber->layer(1)->toGlobal(lp1));
    auto gp2(cscChamber->layer(1)->toGlobal(lp2));
    auto eta1(gp1.eta());
    auto eta2(gp2.eta());
    if (eta1 < eta2)
      std::swap(eta1,eta2);
    std::cout << "{" << i << ", " << eta1 << ", " << eta2 << "},"<< std::endl;
    
    
//     Std ::cout << "WG "<< i << std::endl;
//    wireGroupGEMRollMap_[i] = assignGEMRoll(gp.eta());
  }

//   // print-out
//   for(auto it = wireGroupGEMRollMap_.begin(); it != wireGroupGEMRollMap_.end(); it++) {
//     std::cout << "WG "<< it->first << " GEM pad " << it->second << std::endl;
//   }
*/
