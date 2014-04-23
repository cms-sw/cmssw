#ifndef RecoBTag_TrackIPProducer
#define RecoBTag_TrackIPProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class HistogramProbabilityEstimator;

class TrackIPProducer : public edm::stream::EDProducer<> {
   public:
      explicit TrackIPProducer(const edm::ParameterSet&);
      ~TrackIPProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
    void  checkEventSetup(const edm::EventSetup & iSetup);

    const edm::ParameterSet& m_config;
    edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
    edm::EDGetTokenT<reco::JetTracksAssociationCollection> token_associator;

    bool m_computeProbabilities;
    bool m_computeGhostTrack;
    double m_ghostTrackPriorDeltaR;
    std::auto_ptr<HistogramProbabilityEstimator> m_probabilityEstimator;
    unsigned long long  m_calibrationCacheId2D; 
    unsigned long long  m_calibrationCacheId3D;
    bool m_useDB;

    int  m_cutPixelHits;
    int  m_cutTotalHits;
    double  m_cutMaxTIP;
    double  m_cutMinPt;
    double  m_cutMaxChiSquared;
    double  m_cutMaxLIP;
    bool  m_directionWithTracks;
    bool  m_directionWithGhostTrack;
    bool  m_useTrackQuality;
};
#endif

