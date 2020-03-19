#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxPacker.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLBPacker.h"

void RPCTwinMuxPacker::getRPCTwinMuxRecords(
    RPCInverseLBLinkMap const& lb_map,
    RPCInverseAMCLinkMap const& amc_map,
    int min_bx,
    int max_bx,
    unsigned int bcn,
    RPCDigiCollection const& digis,
    std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >& amc_bx_tmrecord,
    bool ignore_eod) {
  std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > > mlb_bx_lbrecord;
  RPCLBPacker::getRPCLBRecords(lb_map, min_bx, max_bx, bcn, digis, mlb_bx_lbrecord, ignore_eod);

  for (std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > >::const_iterator mlb_bx_lbrecord_it =
           mlb_bx_lbrecord.begin();
       mlb_bx_lbrecord_it != mlb_bx_lbrecord.end();
       ++mlb_bx_lbrecord_it) {
    // multimap, but no splitting for TwinMux inputs
    RPCInverseAMCLinkMap::map_type::const_iterator amc_it(amc_map.getMap().find(mlb_bx_lbrecord_it->first));
    if (amc_it == amc_map.getMap().end()) {
      continue;
    }

    RPCAMCLink amc_id(amc_it->second);
    int amc_input(amc_id.getAMCInput());
    amc_id.setAMCInput();
    std::vector<std::pair<int, rpctwinmux::RPCRecord> >& bx_tmrecord(amc_bx_tmrecord[amc_id]);
    std::vector<std::pair<int, rpctwinmux::RPCRecord> >::iterator tmrecord_it(bx_tmrecord.begin());
    for (std::vector<std::pair<int, RPCLBRecord> >::const_iterator bx_lbrecord = mlb_bx_lbrecord_it->second.begin();
         bx_lbrecord != mlb_bx_lbrecord_it->second.end();
         ++bx_lbrecord) {
      // find the first record at this bx for this amc without this input
      for (; tmrecord_it != bx_tmrecord.end() && tmrecord_it->first < bx_lbrecord->first; ++tmrecord_it)
        ;
      if (tmrecord_it == bx_tmrecord.end() || tmrecord_it->first != bx_lbrecord->first) {
        tmrecord_it = bx_tmrecord.insert(
            tmrecord_it, std::pair<int, rpctwinmux::RPCRecord>(bx_lbrecord->first, rpctwinmux::RPCRecord()));
        tmrecord_it->second.setBXOffset(bx_lbrecord->first);
      }
      rpctwinmux::RPCLinkRecord tm_link_record;
      tm_link_record.setAcknowledge(true);
      tm_link_record.setEOD(bx_lbrecord->second.isEOD());
      tm_link_record.setDelay(bx_lbrecord->second.getDelay());
      tm_link_record.setLinkBoard(bx_lbrecord->second.getLinkBoard());
      tm_link_record.setConnector(bx_lbrecord->second.getConnector());
      tm_link_record.setPartition(bx_lbrecord->second.getPartition());
      tm_link_record.setPartitionData(bx_lbrecord->second.getPartitionData());

      tmrecord_it->second.setRPCLinkRecord(amc_input, tm_link_record);
      // make sure we don't fill this input twice if ignore_eod == true
      ++tmrecord_it;
    }
  }
}
