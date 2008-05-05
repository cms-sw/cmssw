#ifndef MUONCANDPRODUCER_MON_H
#define MUONCANDPRODUCER_MON_H

/*\class MuonCandProducerMon
 *\description Creates full regional muon candidates
               CSC: from l1track objects provided by tf unpacker
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

class MuonCandProducerMon : public edm::EDProducer {

public:

  explicit MuonCandProducerMon(const edm::ParameterSet&);
  ~MuonCandProducerMon();
  
private:

  virtual void beginJob(const edm::EventSetup&);
  virtual void produce (edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  int verbose() {return verbose_;}

 private:
 
  int verbose_;
  edm::InputTag CSCinput_;
  CSCTFPtLUT *ptLUT_;

};

#endif
