#include "PhysicsTools/Heppy/interface/ReclusterJets.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//using namespace std;
using namespace fastjet;

namespace heppy{

ReclusterJets::ReclusterJets(const std::vector<LorentzVector> & objects, double ktpower, double rparam) : 
    ktpower_(ktpower), rparam_(rparam) 
{
  // define jet inputs
  fjInputs_.clear();
  int index=0;
  for (const LorentzVector &o : objects) {
    fastjet::PseudoJet j(o.Px(),o.Py(),o.Pz(),o.E());
    j.set_user_index(index); index++; // in case we want to know which piece ended where
    fjInputs_.push_back(j);
  }

  // choose a jet definition
  fastjet::JetDefinition jet_def;

  // prepare jet def 
  if (ktpower_ == 1.0) {
    jet_def = JetDefinition(kt_algorithm, rparam_);
  }  else if (ktpower_ == 0.0) {
    jet_def = JetDefinition(cambridge_algorithm, rparam_);
  }  else if (ktpower_ == -1.0) {
    jet_def = JetDefinition(antikt_algorithm, rparam_);
  }  else {
    throw cms::Exception("InvalidArgument", "Unsupported ktpower value");
  }
  
  // print out some infos
  //  cout << "Clustering with " << jet_def.description() << endl;
  ///
  // define jet clustering sequence
  fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, jet_def)); 
}

std::vector<math::XYZTLorentzVector> ReclusterJets::makeP4s(const std::vector<fastjet::PseudoJet> &jets) {
  std::vector<math::XYZTLorentzVector> JetObjectsAll;
  for (const fastjet::PseudoJet & pj : jets) {
    JetObjectsAll.push_back( LorentzVector( pj.px(), pj.py(), pj.pz(), pj.e() ) );
  }
  return JetObjectsAll;
}
std::vector<math::XYZTLorentzVector> ReclusterJets::getGrouping(double ptMin) {
  // recluster jet
  inclusiveJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(ptMin));
  // return
  return makeP4s(inclusiveJets_);
}

std::vector<math::XYZTLorentzVector> ReclusterJets::getGroupingExclusive(double dcut) {
  // recluster jet
  exclusiveJets_ = fastjet::sorted_by_pt(fjClusterSeq_->exclusive_jets(dcut));
  // return
  return makeP4s(exclusiveJets_);
}

std::vector<math::XYZTLorentzVector> ReclusterJets::getGroupingExclusive(int njets) {
  // recluster jet
  exclusiveJets_ = fastjet::sorted_by_pt(fjClusterSeq_->exclusive_jets(njets));
  // return
  return makeP4s(exclusiveJets_);
}
}
