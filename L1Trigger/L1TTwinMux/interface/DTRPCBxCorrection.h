//-------------------------------------------------
//
//   Class: DTRPCBxCorrection
//
//   DTRPCBxCorrection
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//--------------------------------------------------

#ifndef DTRPCBxCorrection_H
#define DTRPCBxCorrection_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "CondFormats/L1TObjects/interface/L1TwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TwinMuxParamsRcd.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

class DTRPCBxCorrection  {
public:
  DTRPCBxCorrection(L1MuDTChambPhContainer , L1MuDTChambPhContainer );
  ~DTRPCBxCorrection() {};

  void run(const edm::EventSetup& c);

 edm::ESHandle< L1TwinMuxParams > tmParamsHandle;

 ///Return Output PhContainer
 L1MuDTChambPhContainer getDTContainer(){  return m_dt_tsshifted;}

  static int noRPCHits(L1MuTMChambPhContainer inCon, int bx, int wh, int sec, int st);
 static int noRPCHits(L1MuDTChambPhContainer inCon, int bx, int wh, int sec, int st);
  static int deltaPhi(int dt_phi, int rpc_strip );

private:
  int sign(float);
  int flipBit(int);
  void BxCorrection(int track_seg);

  std::vector<int> concat_delta(std::vector<int>, std::vector<int>, std::vector<int>);

  L1MuTMChambPhContainer m_phiDTDigis;
  L1MuTMChambPhContainer m_phiRPCDigis;
  L1MuDTChambPhContainer m_dt_tsshifted;

  std::vector<L1MuDTChambPhDigi> m_l1ttma_out;

  int m_QualityLimit;
  int m_DphiWindow;

};
#endif
