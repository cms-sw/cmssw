//-------------------------------------------------
//
//   Class: RPCHitCleaner
//
//   RPCHitCleaner
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//--------------------------------------------------

#ifndef L1T_TwinMuxRPC_HITCLEANER_H
#define L1T_TwinMuxRPC_HITCLEANER_H

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

class RPCHitCleaner {
public:
  RPCHitCleaner(RPCDigiCollection const& inrpcDigis);

  void run();

  ///Return Output RPCCollection
  RPCDigiCollection const& getRPCCollection() { return m_outrpcDigis; }

  struct detId_Ext {
    RPCDetId detid;
    int bx;
    int strip;
    bool const operator<(const detId_Ext& o) const {
      return strip < o.strip || (strip == o.strip && detid < o.detid) ||
             (bx < o.bx && strip == o.strip && detid == o.detid);
    }
  };

private:
  ///Input
  RPCDigiCollection const& m_inrpcDigis;
  ///Output
  RPCDigiCollection m_outrpcDigis;
};
#endif
