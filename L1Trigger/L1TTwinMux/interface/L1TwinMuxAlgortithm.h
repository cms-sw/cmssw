//-------------------------------------------------
//
//   Class: L1TwinMuxAlgortithm
//
//   L1TwinMuxAlgortithm
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//--------------------------------------------------

#ifndef L1_TwinMuxAlgorithm_H
#define L1_TwinMuxAlgorithm_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "CondFormats/L1TObjects/interface/L1TwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TwinMuxParamsRcd.h"
#include  "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

class L1TwinMuxAlgortithm  {
public:
  L1TwinMuxAlgortithm()  {};
  ~L1TwinMuxAlgortithm() {};

  void run( edm::Handle<L1MuDTChambPhContainer> phiDigis,
                                                 edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                                 edm::Handle<RPCDigiCollection> rpcDigis,
                                                 const edm::EventSetup& c);

 ///Return Output PhContainer
 L1MuDTChambPhContainer get_ph_tm_output(){  return m_tm_phi_output;}

private:
  int radialAngle(RPCDetId , const edm::EventSetup& , int);
  ///Output PhContainer
  L1MuDTChambPhContainer m_tm_phi_output;

  ///Event Setup Handler
  edm::ESHandle< L1TwinMuxParams > tmParamsHandle;
};
#endif
