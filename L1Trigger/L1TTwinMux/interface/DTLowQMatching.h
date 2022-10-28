//-------------------------------------------------
//
//   Class: DTLowQMatching
//
//   DTLowQMatching
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//   mod.: g Karathanasis
//--------------------------------------------------

#ifndef L1T_TwinMux_DTLowQMatching_H
#define L1T_TwinMux_DTLowQMatching_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TTwinMuxParamsRcd.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

class DTLowQMatching {
public:
  DTLowQMatching(L1MuDTChambPhContainer const*, L1MuDTChambPhContainer const&);

  void run(const L1TTwinMuxParams&);

  static int noRPCHits(L1MuDTChambPhContainer inCon, int bx, int wh, int sec, int st);

private:
  int deltaPhi(int dt_phi, int rpc_strip);

  void Matching(int track_seg);

  L1MuDTChambPhContainer const* m_phiDTDigis;
  L1MuDTChambPhContainer const& m_phiRPCDigis;
  //  L1MuDTChambPhContainer m_phiRPCDigis2;

  int m_DphiWindow;
};
#endif
