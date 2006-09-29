#ifndef JetProducers_TauJet_h
#define JetProducers_TauJet_h

/* Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "JetMETCorrections/TauJet/interface/JetCalibratorTauJet.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class TauJet : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit TauJet (const edm::ParameterSet& ps);

    virtual ~TauJet () {}

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    JetCalibratorTauJet mAlgorithm;
    edm::InputTag mInput;
    std::string mTag;
    int mTauTriggerType;
  };
}


#endif
