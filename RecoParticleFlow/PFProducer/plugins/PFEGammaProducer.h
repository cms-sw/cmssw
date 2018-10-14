#ifndef RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_
#define RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include <memory>

#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgo.h"

class PFEnergyCalibrationHF;
class PFEnergyCalibration;
class PFSCEnergyCalibration;
class PFEnergyCalibrationHF;
class GBRForest;

/**\class PFEGammaProducer
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/


class PFEGammaProducer : public edm::stream::EDProducer<edm::GlobalCache<pfEGHelpers::HeavyObjectCache> > {

 public:
  explicit PFEGammaProducer(const edm::ParameterSet&, const pfEGHelpers::HeavyObjectCache* );
  ~PFEGammaProducer() override {}

  static std::unique_ptr<pfEGHelpers::HeavyObjectCache>
    initializeGlobalCache( const edm::ParameterSet& conf ) {
       return std::unique_ptr<pfEGHelpers::HeavyObjectCache>(new pfEGHelpers::HeavyObjectCache(conf));
   }

  static void globalEndJob(pfEGHelpers::HeavyObjectCache const* ) {}

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  void setPFEGParameters(PFEGammaAlgo::PFEGConfigInfo&);

  void setPFVertexParameters(bool useVertex, const reco::VertexCollection*  primaryVertices);

  void createSingleLegConversions(reco::PFCandidateEGammaExtraCollection &extras,
                                  reco::ConversionCollection &oneLegConversions,
                                  const edm::RefProd<reco::ConversionCollection> &convProd);


  const edm::EDGetTokenT<reco::PFBlockCollection>            inputTagBlocks_;
  const edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation> eetopsSrc_;
  const edm::EDGetTokenT<reco::VertexCollection>             vertices_;

  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  std::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_;

  /// Variables for PFEGamma
  std::string mvaWeightFileEleID_;
  std::vector<double> setchi2Values_;
  double mvaEleCut_;
  bool usePFElectrons_;
  bool usePFPhotons_;
  bool applyCrackCorrectionsElectrons_;
  bool usePFSCEleCalib_;
  bool useEGElectrons_;
  bool useEGammaSupercluster_;
  double sumEtEcalIsoForEgammaSC_barrel_;
  double sumEtEcalIsoForEgammaSC_endcap_;
  double coneEcalIsoForEgammaSC_;
  double sumPtTrackIsoForEgammaSC_barrel_;
  double sumPtTrackIsoForEgammaSC_endcap_;
  double coneTrackIsoForEgammaSC_;
  unsigned int nTrackIsoForEgammaSC_;

  reco::Vertex primaryVertex_;

  std::unique_ptr<reco::PFCandidateCollection>            egCandidates_;
  std::unique_ptr<reco::PFCandidateEGammaExtraCollection> egExtra_;
  std::unique_ptr<reco::ConversionCollection>             singleLegConv_;
  std::unique_ptr<reco::SuperClusterCollection>           sClusters_;

  /// particle flow algorithm
  std::unique_ptr<PFEGammaAlgo>      pfeg_;

  std::string ebeeClustersCollection_;
  std::string esClustersCollection_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGammaProducer);

#endif
