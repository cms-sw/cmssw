#include "EventFilter/RPCRawToDigi/interface/RPCLBPacker.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include "CondFormats/RPCObjects/interface/RPCInverseLBLinkMap.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

void RPCLBPacker::getRPCLBRecords(RPCInverseLBLinkMap const& lb_map,
                                  int min_bx,
                                  int max_bx,
                                  unsigned int bcn,
                                  RPCDigiCollection const& digis,
                                  std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > >& mlb_bx_lbrecord,
                                  bool ignore_eod) {
  if (max_bx - min_bx >= 128 || min_bx < -3564) {  // limit 7 bits RPCLBRecord BCN; avoid overflow
    throw cms::Exception("RPCLBPacker") << "Out-of-range input for min_bx, max_bx (" << min_bx << ": " << max_bx << ")";
  }

  std::map<RPCLBLink, std::vector<RPCLBRecord> > mlb_lbrecords;

  // digi to record ; setBCN to (bx - min_bx) for easy sorting (avoid -1 = 3563 > 0)
  RPCDigiCollection::DigiRangeIterator digi_range_end(digis.end());
  for (RPCDigiCollection::DigiRangeIterator digi_range = digis.begin(); digi_range != digi_range_end; ++digi_range) {
    RPCDigiCollection::DigiRangeIterator::value_type digi_range_value(*digi_range);
    std::pair<RPCInverseLBLinkMap::map_type::const_iterator, RPCInverseLBLinkMap::map_type::const_iterator> lookup_range(
        lb_map.getMap().equal_range(digi_range_value.first.rawId()));

    for (RPCDigiCollection::const_iterator digi = digi_range_value.second.first; digi != digi_range_value.second.second;
         ++digi) {
      if (digi->bx() < min_bx || digi->bx() > max_bx) {
        continue;
      }

      for (RPCInverseLBLinkMap::map_type::const_iterator link_it = lookup_range.first; link_it != lookup_range.second;
           ++link_it) {
        if (link_it->second.second.hasStrip(digi->strip())) {
          unsigned int channel(link_it->second.second.getChannel(digi->strip()));
          RPCLBLink lb_link(link_it->second.first);
          RPCLBLink mlb_link(lb_link);
          mlb_link.setLinkBoard().setConnector();
          mlb_lbrecords[mlb_link].push_back(RPCLBRecord(digi->bx() - min_bx,
                                                        false,
                                                        lb_link.getLinkBoard(),
                                                        false,
                                                        0,
                                                        lb_link.getConnector(),
                                                        (channel > 8 ? 1 : 0),
                                                        0x01 << ((channel - 1) % 8)));
          break;
        }
      }
    }
  }

  // merge records, set correct bcn and delay, and EOD if necessary
  for (std::map<RPCLBLink, std::vector<RPCLBRecord> >::iterator mlb_lbrecords_it = mlb_lbrecords.begin();
       mlb_lbrecords_it != mlb_lbrecords.end();
       ++mlb_lbrecords_it) {
    std::vector<RPCLBRecord>& input(mlb_lbrecords_it->second);
    std::sort(input.begin(), input.end());

    std::vector<std::pair<int, RPCLBRecord> > bx_lbrecord;
    bx_lbrecord.reserve(input.size());

    RPCLBRecord last_lbrecord(input.front());  // it's not empty by construction
    unsigned int idx(0);
    for (std::vector<RPCLBRecord>::const_iterator input_it = input.begin() + 1; input_it <= input.end(); ++input_it) {
      if (input_it != input.end() && ((last_lbrecord.getRecord() & ~RPCLBRecord::partition_data_mask_) ==
                                      (input_it->getRecord() & ~RPCLBRecord::partition_data_mask_))) {
        last_lbrecord.set(last_lbrecord.getRecord() | input_it->getRecord());
      } else {
        unsigned int last_bcn(last_lbrecord.getBCN());
        if (last_bcn > idx) {
          idx = last_bcn;
        }
        unsigned int delay(idx - last_bcn);
        if (ignore_eod && delay == 8) {
          --idx;
          --delay;
          bx_lbrecord.back().second.setEOD(true);
          last_lbrecord.setEOD(true);
        }
        if (delay < 8) {
          last_lbrecord.setDelay(delay);
          last_bcn = (3564 + bcn + min_bx + idx) % 3564;
          last_lbrecord.setBCN(last_bcn);
          last_lbrecord.setBC0(last_bcn == 0);
          bx_lbrecord.push_back(std::pair<int, RPCLBRecord>(min_bx + idx, last_lbrecord));
          ++idx;
        } else {
          bx_lbrecord.back().second.setEOD(true);
        }
        if (input_it != input.end()) {
          last_lbrecord = *input_it;
        }
      }
    }

    mlb_bx_lbrecord.insert(std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > >::value_type(
        mlb_lbrecords_it->first, bx_lbrecord));
  }
}
