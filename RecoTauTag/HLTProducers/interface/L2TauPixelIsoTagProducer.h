#ifndef L2TauPixelIsoTagProducer_h__
#define L2TauPixelIsoTagProducer_h__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/Jet.h"

/** \class L2TauPixelIsoTagProducer
 * Producer of a JetTagCollection where tag is defined as # of pixel tracks
 * in a tau-style isolation cone centered around jet direction.
 * Notes:
 *   - we don't care if signal tracks exist or not (protection against pixel inefficiency).
 *   - Only tracks that belong to the primary vertex are considered.
 *   - If primary vertex doesn't exist, tau jets are tagged as perfectly isolated.
 *
 * \author Vadim Khotilovich
 */
class L2TauPixelIsoTagProducer : public edm::global::EDProducer<>
{
public:

  explicit L2TauPixelIsoTagProducer(const edm::ParameterSet&);

  ~L2TauPixelIsoTagProducer() {};

  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  const edm::EDGetTokenT<edm::View<reco::Jet> > m_jetSrc_token;
  const edm::EDGetTokenT<reco::VertexCollection> m_vertexSrc_token;
  const edm::EDGetTokenT<reco::TrackCollection> m_trackSrc_token;
  const edm::EDGetTokenT<reco::BeamSpot> m_beamSpotSrc_token;

  const int m_maxNumberPV;

  const float m_trackMinPt;
  const float m_trackMaxDxy;
  const float m_trackMaxNChi2;
  const int   m_trackMinNHits;
  const float m_trackPVMaxDZ;

  const float m_isoCone2Min;
  const float m_isoCone2Max;
};

#endif
