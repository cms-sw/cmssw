//-------------------------------------------------
//
//   Class: AlignTrackSegments
//
//   AlignTrackSegments
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//   mod.: G Karathanasis
//--------------------------------------------------

#ifndef L1T_TwinMux_AlignTrackSegments_H
#define L1T_TwinMux_AlignTrackSegments_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TTwinMuxParamsRcd.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

class AlignTrackSegments  {
public:
  AlignTrackSegments(L1MuDTChambPhContainer inphiDigis);
  ~AlignTrackSegments() {};

  void run(const edm::EventSetup& c);

 ///Return Output PhContainer
 const L1MuDTChambPhContainer & getDTContainer(){  return m_dt_tsshifted;}

private:

  ///Output PhContainer
  L1MuDTChambPhContainer m_dt_tsshifted;
  L1MuDTChambPhContainer m_phiDigis;
};
#endif
