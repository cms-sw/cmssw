#ifndef RecoParticleFlow_PFProducer_PFProducer_h_
#define RecoParticleFlow_PFProducer_PFProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

class PFAlgo;
class PFEnergyCalibrationHF;
class GBRForest;

/**\class PFProducer 
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/

class PFProducer : public edm::stream::EDProducer<> {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  ~PFProducer() override;
  
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;

 private:
  edm::EDGetTokenT<reco::PFBlockCollection>  inputTagBlocks_;
  edm::EDGetTokenT<reco::MuonCollection>     inputTagMuons_;
  edm::EDGetTokenT<reco::VertexCollection>   vertices_;
  edm::EDGetTokenT<reco::GsfElectronCollection> inputTagEgammaElectrons_;


  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection> >  inputTagCleanedHF_;
  std::string electronOutputCol_;
  std::string electronExtraOutputCol_;
  std::string photonExtraOutputCol_;

  // NEW EGamma Filters
  edm::EDGetTokenT<edm::ValueMap<reco::GsfElectronRef> >inputTagValueMapGedElectrons_;
  edm::EDGetTokenT<edm::ValueMap<reco::PhotonRef> > inputTagValueMapGedPhotons_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate> > inputTagPFEGammaCandidates_;

  bool use_EGammaFilters_;


  //Use of HO clusters and links in PF Reconstruction
  bool useHO_;

  /// verbose ?
  bool  verbose_;

  // Post muon cleaning ?
  bool postMuonCleaning_;

  // Use PF electrons ?
  bool usePFElectrons_;

  // Use PF photons ?
  bool usePFPhotons_;
  
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
  std::string calibrationsLabel_;

  bool postHFCleaning_;
  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;
  
  /// particle flow algorithm
  std::auto_ptr<PFAlgo>      pfAlgo_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFProducer);

#endif
