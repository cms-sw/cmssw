#ifndef MuonIsolationProducers_MuIsoDepositCopyProducer_H
#define MuonIsolationProducers_MuIsoDepositCopyProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class MuIsoDepositCopyProducer : public edm::stream::EDProducer<> {

public:

  //! constructor
  MuIsoDepositCopyProducer(const edm::ParameterSet&);

  //! destructor
  virtual ~MuIsoDepositCopyProducer();

  //! data making method
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  //! module configuration
  edm::ParameterSet theConfig;

  //! for backward compatibility: take one input module and 
  std::vector<edm::InputTag> theInputTags;
  std::vector<edm::EDGetTokenT<reco::IsoDepositMap> > theInputTokens;
  std::vector<std::string> theDepositNames;

};
#endif
