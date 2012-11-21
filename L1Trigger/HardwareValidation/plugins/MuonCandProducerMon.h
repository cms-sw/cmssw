#ifndef MUONCANDPRODUCER_MON_H
#define MUONCANDPRODUCER_MON_H

/*\class MuonCandProducerMon
 *\description Creates full regional muon candidates
 *             CSCTF: from l1track provided by tf unpacker
 *             DTTF: from L1MuDTTrackContainer by tf unpacker
 *\author N.Leonardo, K.Kotov
 *\date 08.04
 */

// common/system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

class MuonCandProducerMon : public edm::EDProducer {

public:

  explicit MuonCandProducerMon(const edm::ParameterSet&);
  ~MuonCandProducerMon();

private:
  virtual void beginJob(void) {};
  //virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void produce (edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  int verbose() {return verbose_;}

 private:

  int verbose_;
  edm::InputTag CSCinput_;
  edm::InputTag DTinput_;
  CSCTFPtLUT *cscPtLUT_;
  unsigned long long m_scalesCacheID ;
  unsigned long long m_ptScaleCacheID ;

};

#endif
