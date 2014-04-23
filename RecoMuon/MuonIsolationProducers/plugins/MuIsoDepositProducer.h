#ifndef MuonIsolationProducers_MuIsoDepositProducer_H
#define MuonIsolationProducers_MuIsoDepositProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class MuIsoDepositProducer : public edm::stream::EDProducer<> {

public:

  //! constructor
  MuIsoDepositProducer(const edm::ParameterSet&);

  //! destructor
  virtual ~MuIsoDepositProducer();

  //! data making method
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  //! module configuration
  edm::ParameterSet theConfig;

  //! input type. Choose from:
  //! 
  std::string theInputType;

  bool theExtractForCandidate;

  std::string theMuonTrackRefType;
  edm::InputTag theMuonCollectionTag;
  std::vector<std::string> theDepositNames;
  bool theMultipleDepositsFlag;
  reco::isodeposit::IsoDepositExtractor * theExtractor;

};
#endif
