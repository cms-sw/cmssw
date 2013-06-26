#ifndef L2TauPixelIsoTagProducer_h__
#define L2TauPixelIsoTagProducer_h__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"


/** \class L2TauPixelIsoTagProducer
 * Producer of a JetTagCollection where tag is defined as # of pixel tracks
 * in a tau-style isolation cone centered around jet direction.
 * Notes:
 *   - we don't care if signal tracks exist or not (protection against pixel inefficiency).
 *   - Only tracks that belong to the primary vertex are considered.
 *   - If primary vertex doesn't exist, tau jets are tagged as perfectly isolated.
 *
 * \author Vadim Khotilovich
 * $Id: L2TauPixelIsoTagProducer.h,v 1.2 2013/02/27 13:28:15 muzaffar Exp $
 */
class L2TauPixelIsoTagProducer : public edm::EDProducer
{
public:

  explicit L2TauPixelIsoTagProducer(const edm::ParameterSet&);

  ~L2TauPixelIsoTagProducer() {};

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:

  edm::InputTag m_jetSrc;
  edm::InputTag m_vertexSrc;
  edm::InputTag m_trackSrc;
  edm::InputTag m_beamSpotSrc;

  int m_maxNumberPV;

  float m_trackMinPt;
  float m_trackMaxDxy;
  float m_trackMaxNChi2;
  int   m_trackMinNHits;
  float m_trackPVMaxDZ;

  float m_isoCone2Min;
  float m_isoCone2Max;
};

#endif
