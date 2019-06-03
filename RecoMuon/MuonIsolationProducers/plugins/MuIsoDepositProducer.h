#ifndef MuonIsolationProducers_MuIsoDepositProducer_H
#define MuonIsolationProducers_MuIsoDepositProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include <string>

namespace edm {
  class Event;
}
namespace edm {
  class EventSetup;
}

class MuIsoDepositProducer : public edm::stream::EDProducer<> {
public:
  //! constructor
  MuIsoDepositProducer(const edm::ParameterSet&);

  //! destructor
  ~MuIsoDepositProducer() override;

  //! data making method
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //! input type. Choose from:
  //!
  std::string theInputType;

  bool theExtractForCandidate;

  std::string theMuonTrackRefType;
  edm::EDGetToken theMuonCollectionTag;
  std::vector<std::string> theDepositNames;
  bool theMultipleDepositsFlag;
  std::unique_ptr<reco::isodeposit::IsoDepositExtractor> theExtractor;
};
#endif
