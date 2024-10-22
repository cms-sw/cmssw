//-------------------------------------------------
//
//   Class: RPCtoDTTranslator
//
//   RPCtoDTTranslator
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//--------------------------------------------------

#ifndef L1T_TwinMux_RPC_DTTranslator_H
#define L1T_TwinMux_RPC_DTTranslator_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

class RPCtoDTTranslator {
public:
  RPCtoDTTranslator(const RPCDigiCollection& inrpcDigis);

  void run(const RPCGeometry&);

  ///Return Output PhContainer
  L1MuDTChambPhContainer const& getDTContainer() const { return m_rpcdt_translated; }
  L1MuDTChambPhContainer const& getDTRPCHitsContainer() const { return m_rpchitsdt_translated; }

  static int radialAngle(RPCDetId, const RPCGeometry&, int);
  static int bendingAngle(int, int, int);
  //static int bendingAngle(int);
  static int localX(RPCDetId, const RPCGeometry&, int);
  static int localXX(int, int, int);

private:
  ///Output PhContainer
  L1MuDTChambPhContainer m_rpcdt_translated;
  L1MuDTChambPhContainer m_rpchitsdt_translated;

  const RPCDigiCollection& m_rpcDigis;
};
#endif
