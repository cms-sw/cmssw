#ifndef CSCTriggerPrimitives_CSCMotherboardME3141_h
#define CSCTriggerPrimitives_CSCMotherboardME3141_h

/** \class CSCMotherboardME11
 *
 * Extended CSCMotherboardME3141 for ME3141 TMB upgrade
 *
 * \author Sven Dildick March 2014
 *
 * Based on CSCMotherboardME3141 code
 *
 */

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

class CSCGeometry;
class CSCChamber;
class RPCGeometry;

class CSCMotherboardME3141 : public CSCMotherboard
{
 public:
  /** Normal constructor. */
  CSCMotherboardME3141(unsigned endcap, unsigned station, unsigned sector, 
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Default destructor. */
  ~CSCMotherboardME3141();

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc, 
           const CSCComparatorDigiCollection* compdc, 
           const RPCDigiCollection* rpcDigis);

  /// set CSC and RPC geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }

  // check that the RE31 and RE41 chambers are really there
  bool hasRE31andRE41();

 private: 

  const CSCGeometry* csc_g;
  const RPCGeometry* rpc_g;

  std::vector<CSCALCTDigi> alctV;
  std::vector<CSCCLCTDigi> clctV;

  // central LCT bx number 
  int lct_central_bx;

  bool drop_used_clcts;

  // masterswitch
  bool runME3141ILT_;

  // debug rpc matching
  bool debugRPCMatching_;

  //  deltas used to match to RPC pads
  int maxDeltaBXRPC_;
  int maxDeltaRollRPC_;
  int maxDeltaStripRPC_;

  // drop low quality stubs if they don't have RPCs
  bool dropLowQualityCLCTsNoRPC_;
  bool dropLowQualityALCTsNoRPCs_;
};
#endif
