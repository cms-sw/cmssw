#ifndef RecoParticleFlow_PFClusterProducer_PFClusterAnalyzer_
#define RecoParticleFlow_PFClusterProducer_PFClusterAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

/**\class PFClusterAnalyzer 
\brief test analyzer for PFClusters
*/

class PFClusterAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit PFClusterAnalyzer(const edm::ParameterSet&);

  ~PFClusterAnalyzer() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override {}

private:
  void fetchCandidateCollection(edm::Handle<reco::PFClusterCollection>& c,
                                const edm::InputTag& tag,
                                const edm::Event& iSetup) const;

  /*   void printElementsInBlocks(const reco::PFCluster& cluster, */
  /* 			     std::ostream& out=std::cout) const; */

  /// PFClusters in which we'll look for pile up particles
  const edm::InputTag inputTagPFClusters_;

  /// verbose ?
  const bool verbose_;

  /// print the blocks associated to a given candidate ?
  const bool printBlocks_;
};

#endif
