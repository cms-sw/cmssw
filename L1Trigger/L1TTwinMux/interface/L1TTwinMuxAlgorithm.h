//-------------------------------------------------
//
//   Class: L1TwinMuxAlgortithm
//
//   L1TwinMuxAlgortithm
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//   mpd.: g karathanasis
//--------------------------------------------------

#ifndef L1T_TwinMuxL1TTwinMuxAlgorithm_H
#define L1T_TwinMuxL1TTwinMuxAlgorithm_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

class L1TTwinMuxAlgorithm {
public:
  L1TTwinMuxAlgorithm(){};
  ~L1TTwinMuxAlgorithm(){};

  void run(edm::Handle<L1MuDTChambPhContainer> phiDigis,
           edm::Handle<L1MuDTChambThContainer> thetaDigis,
           edm::Handle<RPCDigiCollection> rpcDigis,
           const L1TTwinMuxParams&,
           const RPCGeometry&);

  ///Return Output PhContainer
  L1MuDTChambPhContainer get_ph_tm_output() { return m_tm_phi_output; }

private:
  ///Output PhContainer
  L1MuDTChambPhContainer m_tm_phi_output;
};
#endif
