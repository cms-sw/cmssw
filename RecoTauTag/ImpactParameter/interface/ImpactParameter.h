#ifndef RecoTauTag_ImpactParameter
#define RecoTauTag_ImpactParameter

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoTauTag/ImpactParameter/interface/ImpactParameterAlgorithm.h"


class ImpactParameter : public edm::EDProducer {
   public:
      explicit ImpactParameter(const edm::ParameterSet&);
      ~ImpactParameter();


      virtual void produce(edm::Event&, const edm::EventSetup&);
 private:
      ImpactParameterAlgorithm* algo;
      std::string jetTrackSrc;
      std::string vertexSrc;
      bool usingVertex;
};
#endif

