#ifndef EventFilter_RPCRawToDigi_RPCTwinMuxPacker_h
#define EventFilter_RPCRawToDigi_RPCTwinMuxPacker_h

#include <map>
#include <vector>

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

class RPCInverseLBLinkMap;
class RPCInverseAMCLinkMap;

class RPCTwinMuxPacker {
public:
  /* https://twiki.cern.ch/twiki/bin/viewauth/CMS/DtUpgradeTwinMux#DT_Trigger_and_DT_readout_payloa TwinMux_uROS_payload_v14.xlsx */
  static void getRPCTwinMuxRecords(
      RPCInverseLBLinkMap const& lb_map,
      RPCInverseAMCLinkMap const& amc_map,
      int min_bx,
      int max_bx,
      unsigned int bcn,
      RPCDigiCollection const& digis,
      std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >& amc_bx_tmrecord,
      bool ignore_eod = false);
};

#endif  // EventFilter_RPCRawToDigi_RPCTwinMuxPacker_h
