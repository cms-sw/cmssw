#include "PhysicsTools/Heppy/interface/Megajet.h"

#include <vector>
#include <math.h>
#include <TLorentzVector.h>

using namespace std;

namespace heppy {

// constructor specifying the association methods
Megajet::Megajet(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
                 vector<float> E_vector, int megajet_association_method) : 
  Object_Px(Px_vector), Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), 
  megajet_meth(megajet_association_method),  status(0) { 

  if(Object_Px.size() < 2) cout << "Error in Megajet: you should provide at least two jets to form Megajets" << endl;
  for(int j=0; j<(int)jIN.size(); ++j) {
    jIN.push_back(TLorentzVector(Object_Px[j],Object_Py[j],Object_Pz[j],Object_E[j]));
  }
}

// constructor without specification of the seed and association methods
// in this case, the latter must be given by calling SetMethod before invoking Combine()
Megajet::Megajet(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
                 vector<float> E_vector) : 
  Object_Px(Px_vector), Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), 
  status(0) { 

  if(Object_Px.size() < 2) cout << "Error in Megajet: you should provide at least two jets to form Megajets" << endl;
  for(int j=0; j<(int)jIN.size(); ++j) {
    jIN.push_back(TLorentzVector(Object_Px[j],Object_Py[j],Object_Pz[j],Object_E[j]));
  }
}

vector<float> Megajet::getAxis1(){
  if (status != 1) {
    this->Combine();
    if( megajet_meth == 1) this->CombineMinMass();
    else if(megajet_meth == 2) this->CombineMinHT();
    else if(megajet_meth == 3) this->CombineMinEnergyMass();
    else if(megajet_meth == 4) this->CombineGeorgi();
    else this->CombineMinMass();
  }
  if(jIN.size() > 1) {
    Axis1[0] = jOUT[0].Px() / jOUT[0].P();
    Axis1[1] = jOUT[0].Py() / jOUT[0].P();
    Axis1[2] = jOUT[0].Pz() / jOUT[0].P();
    Axis1[3] = jOUT[0].P();
    Axis1[4] = jOUT[0].E();
  }
  return Axis1;
}
vector<float> Megajet::getAxis2(){
  if (status != 1) {
    this->Combine();
    if( megajet_meth == 1) this->CombineMinMass();
    else if(megajet_meth == 2) this->CombineMinHT();
    else if(megajet_meth == 3) this->CombineMinEnergyMass();
    else if(megajet_meth == 4) this->CombineGeorgi();
    else this->CombineMinMass();
  }
  if(jIN.size() > 1) {
    Axis2[0] = jOUT[1].Px() / jOUT[1].P();
    Axis2[1] = jOUT[1].Py() / jOUT[1].P();
    Axis2[2] = jOUT[1].Pz() / jOUT[1].P();
    Axis2[3] = jOUT[1].P();
    Axis2[4] = jOUT[1].E();
  }
  return Axis2;
}

void Megajet::Combine() {
  int N_JETS = (int)jIN.size();

  int N_comb = 1;
  for(int i = 0; i < N_JETS; i++){
    N_comb *= 2;
  }
    
  // clear some vectors if method Combine() is called again
  if ( !j1.empty() ) {
    j1.clear();
    j2.clear();
    Axis1.clear();
    Axis2.clear();
  }

  for(int j = 0; j < 5; ++j){
    Axis1.push_back(0);
    Axis2.push_back(0);
  }

  int j_count;
  for(int i = 1; i < N_comb-1; i++){
    TLorentzVector j_temp1, j_temp2;
    int itemp = i;
    j_count = N_comb/2;
    int count = 0;
    while(j_count > 0){
      if(itemp/j_count == 1){
	j_temp1 += jIN[count];
      } else {
	j_temp2 += jIN[count];
      }
      itemp -= j_count*(itemp/j_count);
      j_count /= 2;
      count++;
    }

    j1.push_back(j_temp1);
    j2.push_back(j_temp2);
  }
}

void Megajet::CombineMinMass() {
  double M_min = -1;
  // default value (in case none is found)
  TLorentzVector myJ1 = TLorentzVector(0,0,0,0);
  TLorentzVector myJ2 = TLorentzVector(0,0,0,0);
  for(int i=0; i<(int)j1.size(); i++) {
    double M_temp = j1[i].M2()+j2[i].M2();
    if(M_min < 0 || M_temp < M_min){
      M_min = M_temp;
      myJ1 = j1[i];
      myJ2 = j2[i];
    }
  }
  //  myJ1.SetPtEtaPhiM(myJ1.Pt(),myJ1.Eta(),myJ1.Phi(),0.0);
  //  myJ2.SetPtEtaPhiM(myJ2.Pt(),myJ2.Eta(),myJ2.Phi(),0.0);

  jOUT.clear();
  if(myJ1.Pt() > myJ2.Pt()){
    jOUT.push_back(myJ1);
    jOUT.push_back(myJ2);
  } else {
    jOUT.push_back(myJ2);
    jOUT.push_back(myJ1);
  }
  status=1;
}

void Megajet::CombineMinEnergyMass() {
  double M_min = -1;
  // default value (in case none is found)
  TLorentzVector myJ1 = TLorentzVector(0,0,0,0);
  TLorentzVector myJ2 = TLorentzVector(0,0,0,0);
  for(int i=0; i<(int)j1.size(); i++) {
    double M_temp = j1[i].M2()/j1[i].E()+j2[i].M2()/j2[i].E();
    if(M_min < 0 || M_temp < M_min){
      M_min = M_temp;
      myJ1 = j1[i];
      myJ2 = j2[i];
    }
  }
  
  //  myJ1.SetPtEtaPhiM(myJ1.Pt(),myJ1.Eta(),myJ1.Phi(),0.0);
  //  myJ2.SetPtEtaPhiM(myJ2.Pt(),myJ2.Eta(),myJ2.Phi(),0.0);

  jOUT.clear();
  if(myJ1.Pt() > myJ2.Pt()){
    jOUT.push_back(myJ1);
    jOUT.push_back(myJ2);
  } else {
    jOUT.push_back(myJ2);
    jOUT.push_back(myJ1);
  }
  status=1;
}

void Megajet::CombineGeorgi(){
  double M_max = -10000;
  // default value (in case none is found)
  TLorentzVector myJ1 = TLorentzVector(0,0,0,0);
  TLorentzVector myJ2 = TLorentzVector(0,0,0,0);
  for(int i=0; i<(int)j1.size(); i++) {
    int myBeta = 2;
    double M_temp = (j1[i].E()-myBeta*j1[i].M2()/j1[i].E())+(j2[i].E()-myBeta*j2[i].M2()/j2[i].E());
    if(M_max < -9999 || M_temp > M_max){
      M_max = M_temp;
      myJ1 = j1[i];
      myJ2 = j2[i];
    }
  }
  
  //  myJ1.SetPtEtaPhiM(myJ1.Pt(),myJ1.Eta(),myJ1.Phi(),0.0);
  //  myJ2.SetPtEtaPhiM(myJ2.Pt(),myJ2.Eta(),myJ2.Phi(),0.0);

  jOUT.clear();
  if(myJ1.Pt() > myJ2.Pt()){
    jOUT.push_back(myJ1);
    jOUT.push_back(myJ2);
  } else {
    jOUT.push_back(myJ2);
    jOUT.push_back(myJ1);
  }
  status=1;
}

void Megajet::CombineMinHT() {
  double dHT_min = 999999999999999.0;
  // default value (in case none is found)
  TLorentzVector myJ1 = TLorentzVector(0,0,0,0);
  TLorentzVector myJ2 = TLorentzVector(0,0,0,0);
  for(int i=0; i<(int)j1.size(); i++) {
    double dHT_temp = fabs(j1[i].E()-j2[i].E());
    if(dHT_temp < dHT_min){  
      dHT_min = dHT_temp;
      myJ1 = j1[i];
      myJ2 = j2[i];
    }
  }
  
  jOUT.clear();
  if(myJ1.Pt() > myJ2.Pt()){
    jOUT.push_back(myJ1);
    jOUT.push_back(myJ2);
  } else {
    jOUT.push_back(myJ2);
    jOUT.push_back(myJ1);
  }
  status=1;
}

}
