#ifndef EventFilter_RPCRawToDigi_RPCLBPacker_h
#define EventFilter_RPCRawToDigi_RPCLBPacker_h

#include <map>
#include <vector>

#include "CondFormats/RPCObjects/interface/RPCLBLink.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLBRecord.h"

class RPCInverseLBLinkMap;

class RPCLBPacker {
public:
  /* https://twiki.cern.ch/twiki/bin/viewauth/CMS/DtUpgradeTwinMux#RPC_payload RPC_optical_links_data_format.pdf */
  static void getRPCLBRecords(RPCInverseLBLinkMap const& lb_map,
                              int min_bx,
                              int max_bx,
                              unsigned int bcn,
                              RPCDigiCollection const& digis,
                              std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > >& mlb_bx_lbrecord,
                              bool ignore_eod = false);
};

#endif  // EventFilter_RPCRawToDigi_RPCLBPacker_h
