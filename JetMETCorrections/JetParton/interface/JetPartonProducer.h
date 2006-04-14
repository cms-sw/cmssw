#ifndef JetProducers_JetParton_h
#define JetProducers_JetParton_h

/* Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "JetMETCorrections/JetParton/interface/JetCalibratorJetParton.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class JetParton : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit JetParton (const edm::ParameterSet& ps);

    virtual ~JetParton () {}

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    JetCalibratorJetParton mAlgorithm;
    std::string mInput;
    std::string mTag;
    double mRadius;
    int mMixtureType; 
  };
}


#endif
