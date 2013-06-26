#ifndef RecoParticleFlow_PFProducer_PFProducer_h_
#define RecoParticleFlow_PFProducer_PFProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFAlgo;
class PFEnergyCalibrationHF;
class GBRForest;

/**\class PFProducer 
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/

class PFProducer : public edm::EDProducer {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  ~PFProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run &, const edm::EventSetup &) override;

 private:

  edm::InputTag  inputTagBlocks_;
  edm::InputTag  inputTagMuons_;
  edm::InputTag  vertices_;
  edm::InputTag  inputTagEgammaElectrons_;
  std::vector<edm::InputTag>  inputTagCleanedHF_;
  std::string electronOutputCol_;
  std::string electronExtraOutputCol_;
  std::string photonExtraOutputCol_;

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

  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;
  
  /// particle flow algorithm
  std::auto_ptr<PFAlgo>      pfAlgo_;

};

#endif
