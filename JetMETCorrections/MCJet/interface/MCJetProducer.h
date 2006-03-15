#ifndef JetProducers_MCJet_h
#define JetProducers_MCJet_h

/* Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "JetMETCorrections/MCJet/interface/JetCalibratorMCJet.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class MCJet : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit MCJet (const edm::ParameterSet& ps);

    virtual ~MCJet () {}

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    JetCalibratorMCJet mAlgorithm;
    std::string mInput;
    std::string mTag;
  };
}


#endif
