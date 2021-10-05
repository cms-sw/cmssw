#ifndef CSCRecHitD_CSCJetCandidateProducer_h
#define CSCRecHitD_CSCJetCandidateProducer_h

/** \class CSCJetCandidateProducer 
 *
 * Produces a collection of CSCJetCandidate's (2D CSC RecHits as JetCandidates)
 * \author Martin Kwok
 *
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include <FWCore/Utilities/interface/ESGetToken.h>

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCJetCandidate.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCJetCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit CSCJetCandidateProducer(const edm::ParameterSet& ps);
  ~CSCJetCandidateProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<CSCRecHit2DCollection> cscRechitInputToken_;
  typedef std::vector<reco::CSCJetCandidate> CSCJetCandidateCollection;
};

#endif
