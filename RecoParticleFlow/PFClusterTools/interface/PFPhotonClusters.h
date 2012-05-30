#ifndef RecoParticleFlow_PFClusterTools_PFPhotonClusters_h
#define RecoParticleFlow_PFClusterTools_PFPhotonClusters_h 
#include"DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include"DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
//Package: PFClusterTools:
//Class: PFPhotonClusters:
/*
R&D Class to develop ClusterTools for ECal Energy Resolution. 
So Far Members are used for Regression Based Energy Corrections
Developer: Rishi Patel rpatel@cern.ch
*/
class PFPhotonClusters{
 public:
  PFPhotonClusters(reco::PFClusterRef PFClusterRef);
  void SetSeed();
  void PFCrystalCoor();
  void FillClusterShape();
  void FillClusterWidth();
  //return functions
  std::pair<double, double> GetCrysCoor(){
    std::pair<double, double> crys;
    if(isEB_){
      crys.first=CrysEta_;
      crys.second=CrysPhi_;
    }
    else{
      crys.first=CrysX_;
      crys.second=CrysY_;      
    }
    return crys;
  }
  std::pair<double, double> GetCrysIndex(){
    std::pair<int, int> crysI;
    if(isEB_){
      crysI.first=CrysIEta_;
      crysI.second=CrysIPhi_;
    }
    else{
      crysI.first=CrysIX_;
      crysI.second=CrysIY_;      
    }
    return crysI;
  }
  int EtaCrack(){return CrysIEtaCrack_;}
  double E5x5Element(int i, int j){
    //std::cout<<"i, j "<<i<<" , "<<j<<std::endl;
    double E=0;
    if(abs(i)>2 ||abs(j)>2)return E;
    int ind1=i+2;
    int ind2=j+2;
    E=e5x5_[ind1][ind2];
    //std::cout<<"E "<<E<<std::endl;
    return E;
  }
  double PhiWidth(){return sigphiphi_;}
  double EtaWidth(){return sigetaeta_;}
  double EtaPhiWidth(){return sigetaphi_;}
 private:
  reco::PFClusterRef PFClusterRef_;
  //seed detId, position and axis
  DetId idseed_; 
  math::XYZVector seedPosition_, seedAxis_; 
  bool isEB_;
  //crystal Coordinates and indices:
  float CrysX_, CrysY_, CrysEta_, CrysPhi_;
  int CrysIEta_, CrysIPhi_, CrysIEtaCrack_, CrysIX_, CrysIY_;
  //Cluster Shapes:
  double e5x5_[5][5];
  double sigphiphi_, sigetaeta_, sigetaphi_;
  //ClusterWidth
};
#endif
