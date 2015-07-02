#ifndef RecoLocalMuon_ME0RecHitProducer_h
#define RecoLocalMuon_ME0RecHitProducer_h

/** \class ME0RecHitProducer
 *  Module for ME0RecHit production. 
 *  
 *  $Date: 2014/02/04 10:53:23 $
 *  $Revision: 1.1 $
 *  \author M. Maggim -- INFN Bari
 */


#include <memory>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <cstdlib>
#include <bitset>
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"

#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"

#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitBaseAlgo.h"
#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitAlgoFactory.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"

#include <string>

class ME0RecHitBaseAlgo;

class ME0RecHitProducer : public edm::stream::EDProducer<> {

public:
  /// Constructor
  ME0RecHitProducer(const edm::ParameterSet& config);

  /// Destructor
  virtual ~ME0RecHitProducer();

  // Method that access the EventSetup for each run
  virtual void beginRun(const edm::Run&, const edm::EventSetup& ) override;

  /// The method which produces the rechits
  virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;

private:

  // The label to be used to retrieve ME0 digis from the event

  edm::EDGetTokenT<ME0DigiPreRecoCollection> m_token;

  // The reconstruction algorithm
  ME0RecHitBaseAlgo *theAlgo;
  //   static std::string theAlgoName;
};

#endif

