#ifndef RecoJets_JetProducers_interface_JetSpecific_h
#define RecoJets_JetProducers_interface_JetSpecific_h


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/PFClusterJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/Framework/interface/EventSetup.h"



class CaloSubdetectorGeometry;


namespace reco {

  //______________________________________________________________________________    
  // Helper methods to write out specific types

  /// Make CaloJet specifics. Assumes PseudoJet is made from CaloTowerCandidates
  bool makeSpecific(std::vector<reco::CandidatePtr> const & towers, 
		    const CaloSubdetectorGeometry& towerGeometry,
		    reco::CaloJet::Specific* caloJetSpecific);

  void writeSpecific(reco::CaloJet & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );

  
  /// Make PFlowJet specifics. Assumes PseudoJet is made from ParticleFlowCandidates
  bool makeSpecific(std::vector<reco::CandidatePtr> const & particles, 
		    reco::PFJet::Specific* pfJetSpecific);

  void writeSpecific(reco::PFJet  & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );
  
  
  /// Make GenJet specifics. Assumes PseudoJet is made from HepMCCandidate
  bool makeSpecific(std::vector<reco::CandidatePtr> const & mcparticles, 
		    reco::GenJet::Specific* genJetSpecific);

  void writeSpecific(reco::GenJet  & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );
  
  /// Make TrackJet. Assumes constituents point to tracks, through RecoChargedCandidates.
  void writeSpecific(reco::TrackJet  & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );  

/// Make PFClusterJet. Assumes PseudoJet is made from PFCluster
  void writeSpecific(reco::PFClusterJet  & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );
  
  /// Make BasicJet. Assumes nothing about the jet. 
  void writeSpecific(reco::BasicJet  & jet,
		     reco::Particle::LorentzVector const & p4,
		     reco::Particle::Point const & point, 
		     std::vector<reco::CandidatePtr> const & constituents,
		     edm::EventSetup const & c  );
  
  /// converts eta to the corresponding HCAL subdetector.
  HcalSubdetector hcalSubdetector(int iEta);


}

#endif
