#ifndef JetCorrectionProducer_h
#define JetCorrectionProducer_h

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm 
{
  class ParameterSet;
}

class JetCorrector;

namespace cms 
{
  template<class T>
  class JetCorrectionProducer : public edm::EDProducer {
  public:
    typedef std::vector<T> JetCollection;
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

#include "JetCorrectionProducer.icc"

#endif
