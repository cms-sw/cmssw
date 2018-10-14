#ifndef RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_
#define RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

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
  ~PFEGammaProducer() override;
  
  static std::unique_ptr<pfEGHelpers::HeavyObjectCache> 
    initializeGlobalCache( const edm::ParameterSet& conf ) {
       return std::unique_ptr<pfEGHelpers::HeavyObjectCache>(new pfEGHelpers::HeavyObjectCache(conf));
   }
  
  static void globalEndJob(pfEGHelpers::HeavyObjectCache const* ) {
  }

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  

  void setPFEGParameters(PFEGammaAlgo::PFEGConfigInfo&);  
  
  void setPFVertexParameters(bool useVertex, const reco::VertexCollection*  primaryVertices);
   
  void createSingleLegConversions(reco::PFCandidateEGammaExtraCollection &extras, reco::ConversionCollection &oneLegConversions, const edm::RefProd<reco::ConversionCollection> &convProd);
  
  
  edm::EDGetTokenT<reco::PFBlockCollection>  inputTagBlocks_;
  edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation> eetopsSrc_;
  edm::EDGetTokenT<reco::VertexCollection>  vertices_;

  //Use of HO clusters and links in PF Reconstruction

  /// verbose ?
  bool  verbose_;

  // Use photon regression
  bool usePhotonReg_;
  bool useRegressionFromDB_;
  const GBRForest* ReaderGC_;
  const GBRForest* ReaderLC_;
  const GBRForest* ReaderRes_;
  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  // Take PF cluster calibrations from Global Tag ?
  bool useCalibrationsFromDB_;

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
  
  reco::Vertex       primaryVertex_;
  
  std::unique_ptr<reco::PFCandidateCollection> egCandidates_;
  std::unique_ptr<reco::PFCandidateEGammaExtraCollection> egExtra_;
  std::unique_ptr<reco::ConversionCollection> singleLegConv_;
  std::unique_ptr<reco::SuperClusterCollection> sClusters_;  

  /// the unfiltered electron collection 
    
  
  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;
  
  /// particle flow algorithm
  std::unique_ptr<PFEGammaAlgo>      pfeg_;
  
  std::string ebeeClustersCollection_;
  std::string esClustersCollection_;

};

void PFEGammaProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("blocks",          edm::InputTag("particleFlowBlock"));       // PF Blocks label
  desc.add<edm::InputTag>("EEtoPS_source",   edm::InputTag("particleFlowClusterECAL")); // EE to PS association
  // Use Photon identification in PFAlgo (for now this has NO impact, algo is swicthed off hard-coded
  desc.add<bool>("usePhotonReg",             false);
  desc.add<bool>("useVerticesForNeutral",    true);
  desc.add<bool>("useRegressionFromDB",      true);
  desc.add<bool>("usePFSCEleCalib",          true);
  desc.add<std::vector<double>>("calibPFSCEle_Fbrem_barrel", {
      0.6, 6,                                                 // Range of non constant correction
      -0.0255975, 0.0576727, 0.975442, -0.000546394, 1.26147, // standard parameters
      25,                                                     // pt value for switch to low pt corrections
      -0.02025, 0.04537, 0.9728, -0.0008962, 1.172            // low pt parameters
  });
  desc.add<std::vector<double>>("calibPFSCEle_Fbrem_endcap", {
      0.9, 6.5,                                              // Range of non constant correction
      -0.0692932, 0.101776, 0.995338, -0.00236548, 0.874998, // standard parameters eta < switch value
      1.653,                                                 // eta value for correction switch
      -0.0750184, 0.147000, 0.923165, 0.000474665, 1.10782   // standard parameters eta > switch value
  });
  desc.add<std::vector<double>>("calibPFSCEle_barrel", {
      1.004, -1.536, 22.88, -1.467, // standard
      0.3555, 0.6227, 14.65, 2051,  // parameters
      25,                           // pt value for switch to low pt corrections
      0.9932, -0.5444, 0, 0.5438,   // low pt
      0.7109, 7.645, 0.2904, 0      // parameters
  });
  desc.add<std::vector<double>>("calibPFSCEle_endcap", {
      1.153, -16.5975, 5.668,
     -0.1772, 16.22, 7.326,
     0.0483, -4.068, 9.406
  });
  desc.add<bool>           ("useEGammaSupercluster",            true);
  // allow building of candidates with no input or output supercluster?
  desc.add<bool>           ("produceEGCandsWithNoSuperCluster", false);
  desc.add<double>         ("sumEtEcalIsoForEgammaSC_barrel",   1.);
  desc.add<double>         ("sumEtEcalIsoForEgammaSC_endcap",   2.);
  desc.add<double>         ("coneEcalIsoForEgammaSC",           0.3);
  desc.add<double>         ("sumPtTrackIsoForEgammaSC_barrel",  4.);
  desc.add<double>         ("sumPtTrackIsoForEgammaSC_endcap",  4.);
  desc.add<double>         ("coneTrackIsoForEgammaSC",          0.3);
  desc.add<unsigned int>   ("nTrackIsoForEgammaSC",             2);
  desc.add<double>         ("pf_electron_mvaCut",               -0.1);
  desc.add<edm::FileInPath>("pf_electronID_mvaWeightFile",
      edm::FileInPath("RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_PfElectrons23Jan_IntToFloat.txt"));
  desc.add<bool>           ("pf_electronID_crackCorrection",    false);
  desc.add<edm::FileInPath>("pf_convID_mvaWeightFile",
      edm::FileInPath("RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_pfConversionAug0411.txt"));
  desc.add<double>         ("sumPtTrackIsoForPhoton",           2.0);
  desc.add<double>         ("sumPtTrackIsoSlopeForPhoton",      0.001);
  desc.add<std::string>    ("X0_Map",                           "RecoParticleFlow/PFProducer/data/allX0histos.root");

  desc.add<std::string>    ("pf_locC_mvaWeightFile",           "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFClusterLCorr_14Dec2011.root");
  desc.add<std::string>    ("pf_GlobC_mvaWeightFile",          "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFGlobalCorr_14Dec2011.root");
  desc.add<std::string>    ("pf_Res_mvaWeightFile",            "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFRes_14Dec2011.root");
  // ECAL/HCAL PF cluster calibration : take it from global tag ?
  desc.add<bool>           ("useCalibrationsFromDB",           true);
  desc.add<unsigned>       ("algoType",                        0);
  desc.add<edm::InputTag>  ("vertexCollection",                edm::InputTag("offlinePrimaryVertices"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<double>>("nuclCalibFactors", {0.8, 0.15, 0.5, 0.5, 0.05});
    psd0.add<double>("ptErrorSecondary", 1.0);
    psd0.add<bool>("bCalibPrimary", true);
    psd0.add<bool>("bCorrect", true);
    psd0.add<double>("dptRel_MergedTrack", 5.0);
    psd0.add<double>("dptRel_PrimaryTrack", 10.0);
    desc.add<edm::ParameterSetDescription>("iCfgCandConnector", psd0);
  }
  desc.addUntracked<bool>("verbose",false);
  descriptions.add("particleFlowEGamma", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGammaProducer);

#endif
