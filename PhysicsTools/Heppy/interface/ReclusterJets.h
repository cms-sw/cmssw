#ifndef PhysicsTools_Heppy_ReclusterJets_h
#define PhysicsTools_Heppy_ReclusterJets_h

#include <vector>
#include <iostream>
#include <cmath>
#include <TLorentzVector.h>
#include <TMath.h>
#include "DataFormats/Math/interface/LorentzVector.h"

#include <boost/shared_ptr.hpp>
#include <fastjet/internal/base.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

namespace heppy{
class ReclusterJets {
    
 public:
  typedef math::XYZTLorentzVector LorentzVector;

  ReclusterJets(const std::vector<LorentzVector> & objects, double ktpower, double rparam);

  /// get grouping (inclusive jets)
  std::vector<LorentzVector> getGrouping(double ptMin = 0.0);

  /// get grouping (exclusive jets, until n are left)
  std::vector<LorentzVector> getGroupingExclusive(int njets);
  
  /// get grouping (exclusive jets, up to cut dcut)
  std::vector<LorentzVector> getGroupingExclusive(double dcut);

  /// get pruned 4-vector
  LorentzVector getPruned(double zcut, double rcutFactor) ;

  /// get pruned 4-vector for a given subject (must be called after getGroupingExclusive)
  LorentzVector getPrunedSubjetExclusive(unsigned int isubjet, double zcut, double rcutFactor) ;

  /// get pruned 4-vector for a given subject (must be called after getGroupingInclusive)
  LorentzVector getPrunedSubjetInclusive(unsigned int isubjet, double zcut, double rcutFactor) ;


 private:
  // pack the returns in a fwlite-friendly way
  std::vector<LorentzVector> makeP4s(const std::vector<fastjet::PseudoJet> &jets) ;

  // prune and return in fa fwlite-friendly way
  LorentzVector getPruned(const fastjet::PseudoJet & jet, double zcut, double rcutFactor) ;

  // used to handle the inputs
  std::vector<fastjet::PseudoJet> fjInputs_;        // fastjet inputs

  double ktpower_;
  double rparam_;

  /// fastjet outputs
  typedef boost::shared_ptr<fastjet::ClusterSequence>  ClusterSequencePtr;
  ClusterSequencePtr fjClusterSeq_;    
  std::vector<fastjet::PseudoJet> inclusiveJets_; 
  std::vector<fastjet::PseudoJet> exclusiveJets_; 
};
}
#endif   
 
