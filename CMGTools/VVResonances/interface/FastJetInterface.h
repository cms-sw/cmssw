#ifndef CMGTools_VVresonances_FastJetInterface_h
#define CMGTools_VVresonances_FastJetInterface_h

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
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

namespace cmg{
class FastJetInterface {
    
 public:
  typedef math::XYZTLorentzVector LorentzVector;

  FastJetInterface(const std::vector<LorentzVector> & objects, double ktpower, double rparam);
  FastJetInterface(const std::vector<LorentzVector> & objects, double ktpower, double rparam,int active_area_repeats,double ghost_area,double ghost_eta_max,double rho_eta_max);


  void makeInclusiveJets( double);
  void makeExclusiveJets( int);
  void makeExclusiveJets( double);
  void makeExclusiveJetsUpTo(int);
  void makeSubJets(unsigned int);
  void makeSubJets(unsigned int, double);
  void makeSubJets(unsigned int, int);
  void makeSubJetsUpTo(unsigned int, int);
  void prune(bool,double zcut,double rcutfactor);
  void softDrop(bool,double beta,double zcut,double R0);
  bool  massDropTag( unsigned int, double&,double&);


  std::vector<LorentzVector>  get(bool subjets = false);
  std::vector<unsigned int> getConstituents(bool, unsigned int );
  double getArea(bool, unsigned int );
  std::vector<double> nSubJettiness(unsigned int i, int NMAX = 4,unsigned int measureDefinition = 0,unsigned int axesDefinition = 6, double beta = 1.0, double R0 = 0.8,double Rcutoff = -999.0,double akAxisR0 = -999.0,int nPass=-999); 

 private:
  // pack the returns in a fwlite-friendly way
  std::vector<LorentzVector> makeP4s(const std::vector<fastjet::PseudoJet> &jets) ;

  std::vector<fastjet::PseudoJet> input_; 
  std::vector<fastjet::PseudoJet> jets_; 
  std::vector<fastjet::PseudoJet> subjets_; 
  typedef boost::shared_ptr<fastjet::ClusterSequence>  ClusterSequencePtr;
  ClusterSequencePtr clusterSeq_;    



};
}
#endif   
 
