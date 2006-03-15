#ifndef JetProducers_ToyJetCorrector_h
#define JetProducers_ToyJetCorrector_h

/** Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
    $Id: ToyJetCorrector.h,v 1.2 2006/03/08 20:34:19 fedor Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "RecoJets/JetAlgorithms/interface/ToyJetCorrection.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class ToyJetCorrector : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ToyJetCorrector (const edm::ParameterSet& ps);

    virtual ~ToyJetCorrector () {}

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    ToyJetCorrection mAlgorithm;
    std::string mInput;
  };
}


#endif
