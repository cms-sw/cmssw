#ifndef RecoParticleFlow_PFProducer_PFBlockProducer_h_
#define RecoParticleFlow_PFProducer_PFBlockProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"


/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFBlockProducer : public edm::EDProducer {
 public:

  explicit PFBlockProducer(const edm::ParameterSet&);

  ~PFBlockProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob();

  virtual void beginRun(edm::Run & r, const edm::EventSetup & c);

 private:

  

  edm::InputTag   inputTagRecTracks_;
  edm::InputTag   inputTagGsfRecTracks_;
  edm::InputTag   inputTagConvBremGsfRecTracks_;
  edm::InputTag   inputTagRecMuons_;
  edm::InputTag   inputTagPFNuclear_;
  edm::InputTag   inputTagPFClustersECAL_;
  edm::InputTag   inputTagPFClustersHCAL_;
  edm::InputTag   inputTagPFClustersHFEM_;
  edm::InputTag   inputTagPFClustersHFHAD_;
  edm::InputTag   inputTagPFClustersPS_;
  edm::InputTag   inputTagPFConversions_;
  edm::InputTag   inputTagPFV0_;
  edm::InputTag   inputTagEGPhotons_;
  
  /// verbose ?
  bool   verbose_;

  /// use NuclearInteractions ?
  bool   useNuclear_;

  /// use EG photons ? 
  bool useEGPhotons_;
  
  /// switch on/off Conversions
  bool  useConversions_;  
  
  /// switch on/off Conversions Brem Recovery
  bool   useConvBremGsfTracks_;

  /// switch on/off V0
  bool useV0_;

  /// Particle Flow at HLT ?
  bool usePFatHLT_;

  /// Particle flow block algorithm 
  PFBlockAlgo            pfBlockAlgo_;

};

#endif
