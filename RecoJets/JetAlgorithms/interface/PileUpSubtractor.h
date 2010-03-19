#ifndef __PUSubtractor__
#define __PUSubtractor__

#include <vector>
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class PileUpSubtractor{
  
 public:

  typedef boost::shared_ptr<fastjet::ClusterSequence>        ClusterSequencePtr;
  
  PileUpSubtractor(const edm::ParameterSet& iConfig, 
		   std::vector<edm::Ptr<reco::Candidate> >& input,
		   std::vector<fastjet::PseudoJet>& towers,
		   std::vector<fastjet::PseudoJet>& output);
  ~PileUpSubtractor(){;}

  void setAlgorithm(ClusterSequencePtr& algorithm);
  //  void setAlgorithm(fastjet::ClusterSequence& algorithm);
  void setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void calculatePedestal(std::vector<fastjet::PseudoJet> const & coll);
  void subtractPedestal(std::vector<fastjet::PseudoJet> & coll);
  void calculateOrphanInput(std::vector<fastjet::PseudoJet> & orphanInput);
  void offsetCorrectJets();
  double getPileUpAtTower(const reco::CandidatePtr & in);
  double getPileUpEnergy(int ijet){return jetOffset_[ijet];}
  void calculateJetOffset();

 private:

  // From jet producer
  ClusterSequencePtr               fjClusterSeq_;    // fastjet cluster sequence
  std::vector<edm::Ptr<reco::Candidate> >*       inputs_;          // input candidates
  std::vector<fastjet::PseudoJet>* fjInputs_;        // fastjet inputs
  std::vector<fastjet::PseudoJet>* fjJets_;          // fastjet jets

  // PU subtraction parameters
  bool     reRunAlgo_;
  double   jetPtMin_;
  double                nSigmaPU_;                  // number of sigma for pileup
  double                radiusPU_;                  // pileup radius
  CaloGeometry const *  geo_;                       // geometry
  int                   ietamax_;                   // maximum eta in geometry
  int                   ietamin_;                   // minimum eta in geometry
  std::vector<HcalDetId> allgeomid_;                // all det ids in the geometry
  std::map<int,int>     geomtowers_;                // map of geometry towers to det id
  std::map<int,int>     ntowersWithJets_;           // number of towers with jets
  std::map<int,double>  esigma_;                    // energy sigma
  std::map<int,double>  emean_;                     // energy mean

  std::vector<double>   jetOffset_;

};

#endif
