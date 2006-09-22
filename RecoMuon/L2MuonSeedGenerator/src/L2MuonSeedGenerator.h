#ifndef RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H
#define RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H

// -*- C++ -*-
//
// Package:    L2MuonSeedGenerator
// Class:      L2MuonSeedGenerator
// 
/**\class L2MuonSeedGenerator L2MuonSeedGenerator.cc src/L2MuonSeedGenerator/src/L2MuonSeedGenerator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Adam A Everett
// $Id: L2MuonSeedGenerator.h,v 1.1 2006/09/12 16:30:27 bellan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class TrajectorySeed;

namespace edm {class ParameterSet; class Event; class EventSetup;}

//
// class decleration
//

class L2MuonSeedGenerator : public edm::EDProducer {
 public:

  // Constructor
  explicit L2MuonSeedGenerator(const edm::ParameterSet&);

  // Destructor
  ~L2MuonSeedGenerator();
  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  /// get forward bit (true=forward, false=barrel)
  bool isFwd() const { return readDataField( FWDBIT_START, FWDBIT_LENGTH) == 1; }
  
  /// get RPC bit (true=RPC, false = DT/CSC or matched)
  bool isRPC() const { return readDataField( ISRPCBIT_START, ISRPCBIT_LENGTH) == 1; }
  enum { IDXDTCSC_START=26}; enum { IDXDTCSC_LENGTH = 2}; // Bit  26:27 DT/CSC muon index
  enum { IDXRPC_START=28};   enum { IDXRPC_LENGTH = 2};   // Bit  28:29 RPC muon index
  enum { FWDBIT_START=30};   enum { FWDBIT_LENGTH = 1};   // Bit  30    fwd bit
  enum { ISRPCBIT_START=31}; enum { ISRPCBIT_LENGTH = 1}; // Bit  31    isRPC bit
  // ----------member data ---------------------------
  
  std::vector<TrajectorySeed> theSeeds;
  edm::InputTag source_ ;

  const double theL1MinPt;
  const double theL1MaxEta;
  const double theL1MinQuality;
  
 protected:
  unsigned m_dataWord;                                // muon data word (26 bits) :
  // definition of the bit fields
  
  inline unsigned readDataField(unsigned start, unsigned count) const; 
  
};

unsigned L2MuonSeedGenerator::readDataField(unsigned start, unsigned count) const {
  unsigned mask = ( (1 << count) - 1 ) << start;
  return (m_dataWord & mask) >> start;
}

#endif
