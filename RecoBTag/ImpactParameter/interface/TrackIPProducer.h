#ifndef RecoBTag_TrackIPProducer
#define RecoBTag_TrackIPProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"


class TrackIPProducer : public edm::EDProducer {
   public:
      explicit TrackIPProducer(const edm::ParameterSet&);
      ~TrackIPProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
    const edm::ParameterSet& m_config;
    std::string m_associator;
    std::string m_primaryVertexProducer;
    std::string outputInstanceName_;
    bool m_computeProbabilities;

    int  m_cutPixelHits;
    int  m_cutTotalHits;
    double  m_cutMaxTIP;
    double  m_cutMinPt;
    double  m_cutMaxDecayLen;
    double  m_cutMaxChiSquared;
    double  m_cutMaxLIP;
    double m_cutMaxDistToAxis;

};
#endif

