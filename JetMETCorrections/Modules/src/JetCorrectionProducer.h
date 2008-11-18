#ifndef JetCorrectionProducer_h
#define JetCorrectionProducer_h

/* Generic Jet Corrections producer using JetCorrector services
    F.Ratnikov (UMd)
    Dec. 28, 2006
*/

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  class ParameterSet;
}

class JetCorrector;

namespace cms {
  class JetCorrectionProducer : public edm::EDProducer {
  public:
    explicit JetCorrectionProducer (const edm::ParameterSet& fParameters);
    virtual ~JetCorrectionProducer () {}
    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
    edm::InputTag mInput;
    std::vector <std::string> mCorrectorNames;

    // cache
    std::vector <const JetCorrector*> mCorrectors;
    unsigned long long mCacheId;
    bool mVerbose;
  };
}


#endif
