#ifndef RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_
#define RecoParticleFlow_PFEGammaProducer_PFEGammaProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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



class PFEGammaAlgo;
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


class PFEGammaProducer : public edm::EDProducer {
 public:
  explicit PFEGammaProducer(const edm::ParameterSet&);
  ~PFEGammaProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run &, const edm::EventSetup &) override;

 private:
   
  void setPFEGParameters(double mvaEleCut,
			  std::string mvaWeightFileEleID,
			  bool usePFElectrons,
			  const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
			  const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
			  double sumEtEcalIsoForEgammaSC_barrel,
			  double sumEtEcalIsoForEgammaSC_endcap,
			  double coneEcalIsoForEgammaSC,
			  double sumPtTrackIsoForEgammaSC_barrel,
			  double sumPtTrackIsoForEgammaSC_endcap,
			  unsigned int nTrackIsoForEgammaSC,
			  double coneTrackIsoForEgammaSC,
			  bool applyCrackCorrections,
			  bool usePFSCEleCalib,
			  bool useEGElectrons,
			  bool useEGammaSupercluster,
			  bool usePFPhoton,
			  std::string mvaWeightFileConvID,
			  double mvaConvCut,
			  bool useReg,
			  std::string X0_Map,
			  double sumPtTrackIsoForPhoton,
			  double sumPtTrackIsoSlopeForPhoton			  
			);  
  
  void setPFVertexParameters(bool useVertex,
			     const reco::VertexCollection*  primaryVertices);	  
   
  void setPFPhotonRegWeights(
			     const GBRForest *LCorrForestEB,
			     const GBRForest *LCorrForestEE,
			     const GBRForest *GCorrForestBarrel,
			     const GBRForest *GCorrForestEndcapHr9,
			     const GBRForest *GCorrForestEndcapLr9,
			     const GBRForest *PFEcalResolution
			     );   
  
  edm::InputTag  inputTagBlocks_;
  edm::InputTag  vertices_;
  edm::InputTag  inputTagEgammaElectrons_;

  //Use of HO clusters and links in PF Reconstruction

  /// verbose ?
  bool  verbose_;

  // Use photon regression
  bool usePhotonReg_;
  bool useRegressionFromDB_;
  const GBRForest * ReaderGC_;
  const GBRForest* ReaderLC_;
  const GBRForest* ReaderRes_;
  const GBRForest* ReaderLCEB_;
  const GBRForest* ReaderLCEE_;
  const GBRForest* ReaderGCBarrel_;
  const GBRForest* ReaderGCEndCapHighr9_;
  const GBRForest* ReaderGCEndCapLowr9_;
  const GBRForest* ReaderEcalRes_;
  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  // Take PF cluster calibrations from Global Tag ?
  bool useCalibrationsFromDB_;

  boost::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_;  
  
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
  bool               useVertices_;   
  
  std::auto_ptr< reco::PFCandidateCollection >    egCandidates_;
  std::auto_ptr< reco::CaloClusterCollection >    ebeeClusters_;
  std::auto_ptr< reco::CaloClusterCollection >    esClusters_;
  std::auto_ptr< reco::SuperClusterCollection >    sClusters_;

  /// the unfiltered electron collection 
  std::auto_ptr<reco::PFCandidateEGammaExtraCollection>    egExtra_;  
  
  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;
  
  /// particle flow algorithm
  std::auto_ptr<PFEGammaAlgo>      pfeg_;

};

#endif
