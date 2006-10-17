#ifndef RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H
#define RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H

//-------------------------------------------------
//
/**  \class L2MuonSeedGenerator
 * 
 *   L2 muon seed generator:
 *   Transform the L1 informations in seeds for the
 *   L2 muon reconstruction
 *
 *
 *   $Date: $
 *   $Revision: $
 *
 *   \author  A.Everett, R.Bellan
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class MuonServiceProxy;
class MeasurementEstimator;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L2MuonSeedGenerator : public edm::EDProducer {

 private:

  enum { IDXDTCSC_START=26}; enum { IDXDTCSC_LENGTH = 2}; // Bit  26:27 DT/CSC muon index
  enum { IDXRPC_START=28};   enum { IDXRPC_LENGTH = 2};   // Bit  28:29 RPC muon index
  enum { FWDBIT_START=30};   enum { FWDBIT_LENGTH = 1};   // Bit  30    fwd bit
  enum { ISRPCBIT_START=31}; enum { ISRPCBIT_LENGTH = 1}; // Bit  31    isRPC bit
  
 public:
  
  /// Constructor
  explicit L2MuonSeedGenerator(const edm::ParameterSet&);

  /// Destructor
  ~L2MuonSeedGenerator();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  /// definition of the bit fields
  unsigned readDataField(unsigned start, unsigned count) const; 
  
  /// get forward bit (true=forward, false=barrel)
  bool isFwd() const;
  
  /// get RPC bit (true=RPC, false = DT/CSC or matched)
  bool isRPC() const;

 private:
  edm::InputTag theSource;
  std::string thePropagatorName;

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;

  /// muon data word (26 bits)
  unsigned theDataWord;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  

  MeasurementEstimator *theEstimator;
};

#endif
