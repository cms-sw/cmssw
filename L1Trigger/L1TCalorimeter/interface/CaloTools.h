///
/// \class l1t::CaloTools
///
/// Description: A collection of useful functions for the Calorimeter that are of generic interest
///
/// Implementation:
///   currently implimented as a static class rather than a namespace, open to re-writing it as namespace  
///
/// \author: Sam Harper - RAL
///

//

#ifndef L1Trigger_L1TCommon_CaloTools_h
#define L1Trigger_L1TCommon_CaloTools_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

namespace l1t {

  class CaloTools{
  
    //class is not designed to be instanced
  private:
    CaloTools(){}
    ~CaloTools(){}
  
  public:
 
    enum SubDet{ECAL=0x1,HCAL=0x2,CALO=0x3}; //CALO is a short cut for ECAL|HCAL

    static const l1t::CaloTower& getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi);

    //returns a hash suitable for indexing a vector (note does not check for validity yet of iEta,iPhi)
    static size_t caloTowerHash(int iEta,int iPhi);
    
    //returns the hw Et sum of of a rectangle bounded by iEta-localEtaMin,iEta+localEtaMax,iPhi-localPhiMin,iPhi-localPhiMax (inclusive)
    //sum is either ECAL, HCAL or CALO (ECAL+HCAL) Et
    static int calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			  int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,SubDet etMode=CALO);

    //returns the number of towers with minHwEt<=hwEt<=maxHwEt and iEtaMin<=iEta<=iEtaMax and iPhiMin<=iPhi<=iPhiMax
    //hwEt is either ECAL, HCAL or CALO (ECAL+HCAL) Et
    static size_t calNrTowers(int iEtaMin,int iEtaMax,int iPhiMin,int iPhiMax,const std::vector<l1t::CaloTower>& towers,int minHwEt,int maxHwEt,SubDet etMode=CALO);

  private:
    static const l1t::CaloTower nullTower_; //to return when we need to return a tower which was not found/invalid rather than throwing an exception
  };

}


#endif
