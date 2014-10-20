#ifndef HemisphereViaKt_h
#define HemisphereViaKt_h

#include <vector>
#include <iostream>
#include <cmath>
#include <TLorentzVector.h>
#include <TMath.h>

using namespace std;
using std::vector;
using std::cout;
using std::endl;

#include <boost/shared_ptr.hpp>
#include <fastjet/internal/base.hh>
//#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

//FASTJET_BEGIN_NAMESPACE

class HemisphereViaKt {
    
 public:

  HemisphereViaKt(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector, vector<float> E_vector);
  HemisphereViaKt(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector, vector<float> E_vector, double ktpower);

  // get grouping
  std::vector<vector<float> > getGrouping();


 private:

  // the hemisphere separation algorithm
  int Reconstruct();

  // used to handle the inputs
  vector<float> Object_Px;
  vector<float> Object_Py;
  vector<float> Object_Pz;
  vector<float> Object_E;

  // used to store the exclusive jets
  vector<float> JetObject_Px;
  vector<float> JetObject_Py;
  vector<float> JetObject_Pz;
  vector<float> JetObject_E;
  std::vector<vector<float> > JetObjectAll;

  std::vector<fastjet::PseudoJet> fjInputs_;        // fastjet inputs
  std::vector<fastjet::PseudoJet> fjJets_;          // fastjet jets
  typedef boost::shared_ptr<fastjet::ClusterSequence>  ClusterSequencePtr;
  ClusterSequencePtr fjClusterSeq_;    

  double ktpower_;
  double rparam_;

  int numLoop;
  int ktpower;
  int status;
  int dbg;
    
};

//FASTJET_END_NAMESPACE
 
#endif    
