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
 *   $Date: 2006/10/17 16:09:25 $
 *   $Revision: 1.3 $
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
 
 public:
  
  /// Constructor
  explicit L2MuonSeedGenerator(const edm::ParameterSet&);

  /// Destructor
  ~L2MuonSeedGenerator();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  edm::InputTag theSource;
  std::string thePropagatorName;

  const double theL1MinPt;
  const double theL1MaxEta;
  const unsigned theL1MinQuality;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  

  MeasurementEstimator *theEstimator;
};

#endif
