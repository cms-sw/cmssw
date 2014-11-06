#ifndef PhysicsTools_Heppy_HemisphereViaKt_h
#define PhysicsTools_Heppy_HemisphereViaKt_h

#include <vector>
#include <iostream>
#include <cmath>
#include <TLorentzVector.h>
#include <TMath.h>


#include <boost/shared_ptr.hpp>
#include <fastjet/internal/base.hh>
//#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

//FASTJET_BEGIN_NAMESPACE

namespace heppy {

class HemisphereViaKt {
    
 public:

  HemisphereViaKt(std::vector<float> Px_vector, 
		  std::vector<float> Py_vector, 
		  std::vector<float> Pz_vector, 
		  std::vector<float> E_vector);
  HemisphereViaKt(std::vector<float> Px_vector, 
		  std::vector<float> Py_vector, 
		  std::vector<float> Pz_vector, 
		  std::vector<float> E_vector, 
		  double ktpower);

  // get grouping
  std::vector<std::vector<float> > getGrouping();


 private:

  // the hemisphere separation algorithm
  int Reconstruct();

  // used to handle the inputs
  std::vector<float> Object_Px;
  std::vector<float> Object_Py;
  std::vector<float> Object_Pz;
  std::vector<float> Object_E;

  // used to store the exclusive jets
  std::vector<float> JetObject_Px;
  std::vector<float> JetObject_Py;
  std::vector<float> JetObject_Pz;
  std::vector<float> JetObject_E;
  std::vector<std::vector<float> > JetObjectAll;

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

}
 
#endif    
