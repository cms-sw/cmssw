#ifndef RecoBTag_TrackProbability
#define RecoBTag_TrackProbability

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoBTag/TrackProbability/interface/TrackProbabilityAlgorithm.h"


class TrackProbability : public edm::EDProducer {
   public:
      explicit TrackProbability(const edm::ParameterSet&);
      ~TrackProbability();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
    unsigned long long  m_calibrationCacheId2D; 
    unsigned long long  m_calibrationCacheId3D;
    bool m_useDB;
 
    const edm::ParameterSet& m_config;
    TrackProbabilityAlgorithm m_algo;
    std::string m_associator;
    std::string m_primaryVertexProducer;
    std::string outputInstanceName_;
};
#endif

