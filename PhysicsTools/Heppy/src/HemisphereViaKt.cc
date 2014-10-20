#include "PhysicsTools/Heppy/interface/HemisphereViaKt.h"

//using namespace std;
using namespace fastjet;

using std::vector;
using std::cout;
using std::endl;

HemisphereViaKt::HemisphereViaKt(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
				 vector<float> E_vector) : Object_Px(Px_vector),
							   Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), ktpower(-1), status(0), dbg(1) {
  numLoop =0;
}

HemisphereViaKt::HemisphereViaKt(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
				 vector<float> E_vector, double ktpower) : Object_Px(Px_vector),
									Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), ktpower_(ktpower), rparam_(50.0), status(0), dbg(1) {
  numLoop =0;
}

std::vector<vector<float> > HemisphereViaKt::getGrouping(){
  this->Reconstruct();
  return JetObjectAll;
}

int HemisphereViaKt::Reconstruct(){

  numLoop=0; // initialize numLoop for Zero
  int vsize = (int) Object_Px.size();
  if((int) Object_Py.size() != vsize || (int) Object_Pz.size() != vsize){
    cout << "WARNING!!!!! Input vectors have different size! Fix it!" << endl;
    return 0;
  }


  ///
  // define jet inputs
  fjInputs_.clear();
  for (int i = 0; i <vsize; ++i){
    fastjet::PseudoJet j(Object_Px[i],Object_Py[i],Object_Pz[i],Object_E[i]);
    //if ( fabs(j.rap()) < inputEtaMax )
    fjInputs_.push_back(j);
  }

  // choose a jet definition
  fastjet::JetDefinition jet_def;

  // prepare jet def 
  // rparam_ unused for the exclusive jets  set here to a default lenght
  if (ktpower_ == 1.0) {
    jet_def = JetDefinition(kt_algorithm, rparam_);
  }  else if (ktpower_ == 0.0) {
    jet_def = JetDefinition(cambridge_algorithm, rparam_);
  }  else if (ktpower_ == -1.0) {
    jet_def = JetDefinition(antikt_algorithm, rparam_);
  } 
  
  // print out some infos
  //  cout << "Clustering with " << jet_def.description() << endl;
  ///
  // define jet clustering sequence
  fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, jet_def)); 

  // recluster jet
  // std::vector<fastjet::PseudoJet> inclusiveJets = fastjet::sorted_by_pt( fjClusterSeq_->inclusive_jets());
  std::vector<fastjet::PseudoJet> exclusiveJets = fastjet::sorted_by_pt( fjClusterSeq_->exclusive_jets(2) );


  //  cout << "number of jets " << fjInputs_.size() << endl;
  //  cout << "number of hemispheres " << exclusiveJets.size() << endl;

  for (unsigned int i = 0; i <exclusiveJets.size(); ++i){
    
    JetObject_Px.push_back(exclusiveJets.at(i).px());
    //    cout << "i=" << i << " px=" << exclusiveJets.at(i).px() ;
    JetObject_Py.push_back(exclusiveJets.at(i).py());
    //    cout << " py=" << exclusiveJets.at(i).py() << endl;
    JetObject_Pz.push_back(exclusiveJets.at(i).pz());
    JetObject_E.push_back(exclusiveJets.at(i).e());


  }

  JetObjectAll.push_back(JetObject_Px);
  JetObjectAll.push_back(JetObject_Py);
  JetObjectAll.push_back(JetObject_Pz);
  JetObjectAll.push_back(JetObject_E);

  /*
  for (unsigned int i = 0; i < exclusiveJets.size(); i++ ) {
    std::vector<PseudoJet> constituents = exclusiveJets.at(i).constituents();
    cout << " hemisphere (" << i << ") with N=" << constituents.size() << " jets [ and pt,eta,phi,mass=" << exclusiveJets.at(i).pt() << 
      "," << exclusiveJets.at(i).eta() << ","<< exclusiveJets.at(i).phi() << ","<< exclusiveJets.at(i).m() << "]"<< endl;

    for (unsigned int j=0; j<constituents.size(); j++) {
      cout << "    pt (" << i << ")" << constituents.at(j).pt();
    }
    cout << " " << endl;
  }
  */


  if(JetObjectAll.size()<2) return -1;
  return 1;
    
    
}
