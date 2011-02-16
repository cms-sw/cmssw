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
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void beginRun(edm::Run &, const edm::EventSetup &);

 private:

  edm::InputTag  inputTagBlocks_;
  edm::InputTag  inputTagMuons_;
  edm::InputTag  vertices_;
  edm::InputTag  inputTagEgammaElectrons_;
  std::vector<edm::InputTag>  inputTagCleanedHF_;
  std::string electronOutputCol_;
  std::string electronExtraOutputCol_;

  /// verbose ?
  bool  verbose_;

  // Post muon cleaning ?
  bool postMuonCleaning_;

  // Use PF electrons ?
  bool usePFElectrons_;

  // Use PF photons ?
  bool usePFPhotons_;

  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  
  /// particle flow algorithm
  std::auto_ptr<PFAlgo>      pfAlgo_;

};

#endif
