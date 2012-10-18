  
#include "HFValueStruct.h"
#include <stdio.h>
#include <stdlib.h>
//version -1 will take information from DB (NOT DONE YET)
//version 0 has energy corrections on, everything else off
//version 1 has energy correction, and pile up slope and interceept on
// version 2+ will use defaults of 'do nothing' 
//
//
//


reco::HFValueStruct::HFValueStruct(const int& version, const std::vector<double>& vect): v_(version),hfvv_(vect) {
  //if(v_==-1) hfvv_=SetHfvvFromDB_();
  //v==99 will always give defaults
  
  //version control, add in versions as they appear!!
  if(v_==0 || v_==1) doEnCor_=true;
  else  doEnCor_=false;
  
  if(v_==1) doPU_=true;
  else doPU_=false;
  
  
}



int reco::HFValueStruct::indexByIeta(int& ieta)const{
  return (ieta>0)?(abs(ieta)-29+13):(41-abs(ieta));
}
int reco::HFValueStruct::ietaByIndex(int& indx)const{
  return (indx>13)?(indx+29-13):(indx-41);
}
//version 0
// EnCor=energy corrections,default 1.0, 26 slots

//version 1
//PUSlope= slope for pile up corrections,default 0.0, 26 slots, 0,1,12,13,24,25 are all defaults always

//PUIntercept= intercept  slope for pile up corrections,default 1.0, 26 slots, 0,1,12,13,24,25 are all defaults always




// returns single value by index

double reco::HFValueStruct::EnCor(int ieta)const{
  int indx=indexByIeta(ieta);
  if(doEnCor_) return hfvv_[indx];
  else return 1.0;}
double reco::HFValueStruct::PUSlope(int ieta)const{
  int indx=indexByIeta(ieta)+26;
  if(doPU_) return hfvv_[indx];
  else return 0.0;}
double reco::HFValueStruct::PUIntercept(int ieta)const{
  int indx=indexByIeta(ieta)+52;
  if(doPU_) return hfvv_[indx];
  else return 1.0;}

// sets single value by index
void reco::HFValueStruct::setEnCor(int ieta,double val){
  int indx=indexByIeta(ieta);
  hfvv_[indx]=val;}
void reco::HFValueStruct::setPUSlope(int ieta,double val){
  int indx=indexByIeta(ieta)+26;
  hfvv_[indx]=val;}
void reco::HFValueStruct::setPUIntercept(int ieta,double val){
  int indx=indexByIeta(ieta)+52;
  hfvv_[indx]=val;}



// returns whole vector
std::vector<double> reco::HFValueStruct::EnCor()const{
  std::vector<double> vct;
  if(doEnCor_){
    for(int ii=0;ii<13;ii++)
      vct.push_back(hfvv_[ii]);
  }else{
    for(int ii=0;ii<13;ii++)
      vct.push_back(1.0);
  }
  return vct;}

std::vector<double> reco::HFValueStruct::PUSlope()const{
  std::vector<double> vct;
  if(doPU_){
    for(int ii=0;ii<13;ii++)
      vct.push_back(hfvv_[ii+26]);
  }else{
    for(int ii=0;ii<13;ii++)
      vct.push_back(0.0);
  }
  return vct;}

std::vector<double> reco::HFValueStruct::PUIntercept()const{	
  std::vector<double> vct;
  if(doPU_){
    for(int ii=0;ii<13;ii++)
      vct.push_back(hfvv_[ii+52]);
  }else{
    for(int ii=0;ii<13;ii++)
      vct.push_back(1.0);
  }
  return vct;}

// set whole vector
void reco::HFValueStruct::setEnCor(const std::vector<double>& val){
  for(int ii=0;ii<13;ii++) hfvv_[ii]=val[ii];}
void reco::HFValueStruct::setPUSlope(const std::vector<double>& val){
  for(int ii=0;ii<13;ii++) hfvv_[ii+26]=val[ii];}
void reco::HFValueStruct::setPUIntercept(const std::vector<double>& val){
  for(int ii=0;ii<13;ii++) hfvv_[ii+52]=val[ii];
}
