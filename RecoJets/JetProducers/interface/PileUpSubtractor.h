#ifndef __PUSubtractor__
#define __PUSubtractor__

#include <vector>
#include "boost/shared_ptr.hpp"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/GhostedAreaSpec.hh"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class PileUpSubtractor{
  
 public:

  typedef boost::shared_ptr<fastjet::ClusterSequence>        ClusterSequencePtr;
  typedef boost::shared_ptr<fastjet::GhostedAreaSpec>        ActiveAreaSpecPtr;
  typedef boost::shared_ptr<fastjet::RangeDefinition>        RangeDefPtr;
  typedef boost::shared_ptr<fastjet::JetDefinition>          JetDefPtr;
  
  PileUpSubtractor(const edm::ParameterSet& iConfig,  edm::ConsumesCollector && iC); 
  virtual ~PileUpSubtractor(){;}

virtual void setDefinition(JetDefPtr const & jetDef);
virtual void reset(std::vector<edm::Ptr<reco::Candidate> >& input,
	     std::vector<fastjet::PseudoJet>& towers,
	     std::vector<fastjet::PseudoJet>& output);
virtual void setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup);
virtual void calculatePedestal(std::vector<fastjet::PseudoJet> const & coll);
virtual void subtractPedestal(std::vector<fastjet::PseudoJet> & coll);
virtual void calculateOrphanInput(std::vector<fastjet::PseudoJet> & orphanInput);
virtual void offsetCorrectJets();
virtual double getMeanAtTower(const reco::CandidatePtr & in) const;
virtual double getSigmaAtTower(const reco::CandidatePtr & in) const;
virtual double getPileUpAtTower(const reco::CandidatePtr & in) const;
virtual double getPileUpEnergy(int ijet) const {return jetOffset_[ijet];}
 virtual double getCone(double cone, double eta, double phi, double& et, double& pu);
 int getN(const reco::CandidatePtr & in) const;
 int getNwithJets(const reco::CandidatePtr & in) const;

 int ieta(const reco::CandidatePtr & in) const;
 int iphi(const reco::CandidatePtr & in) const;

 protected:

  // From jet producer
  JetDefPtr                       fjJetDefinition_;  // fastjet jet definition
  ClusterSequencePtr              fjClusterSeq_;    // fastjet cluster sequence
  std::vector<edm::Ptr<reco::Candidate> >*       inputs_;          // input candidates
  std::vector<fastjet::PseudoJet>* fjInputs_;        // fastjet inputs
  std::vector<fastjet::PseudoJet>* fjJets_;          // fastjet jets
  std::vector<fastjet::PseudoJet> fjOriginalInputs_;        // to back-up unsubtracted fastjet inputs

  // PU subtraction parameters
  bool     reRunAlgo_;
  bool     doAreaFastjet_;
  bool     doRhoFastjet_;
  double   jetPtMin_;
  double   puPtMin_;

  double                nSigmaPU_;                  // number of sigma for pileup
  double                radiusPU_;                  // pileup radius
  ActiveAreaSpecPtr               fjActiveArea_;    // fastjet active area definition
  RangeDefPtr                     fjRangeDef_;      // range definition

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

#include "FWCore/PluginManager/interface/PluginFactory.h"
namespace edm {class ParameterSet; class EventSetup; class ConsumesCollector;}
typedef edmplugin::PluginFactory<PileUpSubtractor *(const edm::ParameterSet &, edm::ConsumesCollector &&)> PileUpSubtractorFactory;

#endif
