#ifndef RecoBTag_TrackIPProducer
#define RecoBTag_TrackIPProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class HistogramProbabilityEstimator;

class TrackIPProducer : public edm::EDProducer {
   public:
      explicit TrackIPProducer(const edm::ParameterSet&);
      ~TrackIPProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
    void  checkEventSetup(const edm::EventSetup & iSetup);

    const edm::ParameterSet& m_config;
    edm::InputTag m_associator;
    edm::InputTag m_primaryVertexProducer;
    bool m_computeProbabilities;
    HistogramProbabilityEstimator * m_probabilityEstimator;
    unsigned long long  m_calibrationCacheId2D; 
    unsigned long long  m_calibrationCacheId3D;
    bool m_useDB;


    int  m_cutPixelHits;
    int  m_cutTotalHits;
    double  m_cutMaxTIP;
    double  m_cutMinPt;
    double  m_cutMaxDecayLen;
    double  m_cutMaxChiSquared;
    double  m_cutMaxLIP;
    double  m_cutMaxDistToAxis;
    bool  m_directionWithTracks;
    bool  m_useTrackQuality;
};
#endif

