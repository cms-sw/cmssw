#include <iostream>
#include <ostream>
#include <cmath>

#include "L1Trigger/L1TMuonOverlap/interface/OMTFResult.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

////////////////////////////////////////////
////////////////////////////////////////////
OMTFResult::OMTFResult(){

  clear();

}
////////////////////////////////////////////
////////////////////////////////////////////

void OMTFResult::setRefPhiRHits(unsigned int iRefLayer, int iRefPhiRHit){
      refPhiRHit1D[iRefLayer] = iRefPhiRHit;
}
////////////////////////////////////////////
////////////////////////////////////////////

void OMTFResult::addResult(unsigned int iRefLayer,
			   unsigned int iLayer,
			   unsigned int val,
			   int iRefPhi, 
			   int iRefEta){

  refPhi1D[iRefLayer] = iRefPhi;
  refEta1D[iRefLayer] = iRefEta;
  results[iLayer][iRefLayer] = val;

}
////////////////////////////////////////////
////////////////////////////////////////////
void OMTFResult::clear(){

  results1D.assign(OMTFConfiguration::instance()->nRefLayers,0);
  hits1D.assign(OMTFConfiguration::instance()->nRefLayers,0);
  results.assign(OMTFConfiguration::instance()->nLayers,results1D);
  refPhi1D.assign(OMTFConfiguration::instance()->nRefLayers,1024);
  refEta1D.assign(OMTFConfiguration::instance()->nRefLayers,1024);
  hitsBits.assign(OMTFConfiguration::instance()->nRefLayers,0);  
  refPhiRHit1D.assign(OMTFConfiguration::instance()->nRefLayers,1024);
}
////////////////////////////////////////////
////////////////////////////////////////////
void OMTFResult::finalise(){

  for(unsigned int iLogicLayer=0;iLogicLayer<results.size();++iLogicLayer){

    //unsigned int connectedLayer = OMTFConfiguration::instance()->logicToLogic[iLogicLayer];    
    auto iter = OMTFConfiguration::instance()->logicToLogic.find(iLogicLayer);
    unsigned int connectedLayer = iter->second;

    for(unsigned int iRefLayer=0;iRefLayer<results[iLogicLayer].size();++iRefLayer){
      ///If connected layer (POS or BEND) has not been fired, ignore this layer also
      unsigned int val = results[connectedLayer][iRefLayer]>0 ? results[iLogicLayer][iRefLayer]: 0;
      results1D[iRefLayer]+=val;
      hitsBits[iRefLayer]+=(val>0)*std::pow(2,iLogicLayer);
      ///Do not count bending layers in hit count
      if(!OMTFConfiguration::instance()->bendingLayers.count(iLogicLayer)) hits1D[iRefLayer]+=(val>0);
    }      
  }
}
////////////////////////////////////////////
////////////////////////////////////////////
bool OMTFResult::empty() const{

  unsigned int nHits = 0;

  for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::instance()->nRefLayers;++iRefLayer){
    nHits+=hits1D[iRefLayer];
  }      
  return (nHits==0);
}
////////////////////////////////////////////
////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const OMTFResult & aResult){

 for(unsigned int iLogicLayer=0;iLogicLayer<aResult.results.size();++iLogicLayer){
    out<<"Logic layer: "<<iLogicLayer<<" results: ";
    for(unsigned int iRefLayer=0;iRefLayer<aResult.results[iLogicLayer].size();++iRefLayer){
      out<<aResult.results[iLogicLayer][iRefLayer]<<"\t";
    }
    out<<std::endl;
  }

 out<<"      Sum over layers: ";
 for(unsigned int iRefLayer=0;iRefLayer<aResult.results1D.size();++iRefLayer){
   out<<aResult.results1D[iRefLayer]<<"\t";
 }

 out<<std::endl;

 out<<"       Number of hits: ";
 for(unsigned int iRefLayer=0;iRefLayer<aResult.hits1D.size();++iRefLayer){
   out<<aResult.hits1D[iRefLayer]<<"\t";
 }

  return out;
}
////////////////////////////////////////////
////////////////////////////////////////////
