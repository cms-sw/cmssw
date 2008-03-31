/* L2TauIsolationInfo Class
Holds output of the Tau L2 IsolationProducer
 
Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/



#ifndef L2TAUISOLATION_INFO_H
#define L2TAUISOLATION_INFO_H

#include <iostream>



namespace reco {

class L2TauIsolationInfo
{
 public:

  double ECALIsolConeCut; //ECAL : Isolation Cones cut
  int ECALIsolDiscriminator; //ECAL :Isolation Cones Answer

  double SeedTowerEt;//Seed CaloTower Et
  double TowerIsolConeCut;//ECAL+HCAL : Isolation Cones Cut
  int TowerIsolDiscriminator;//ECAL+HCAL:Isolation Cones Discriminator
  
  int ECALClusterNClusters;//ECAL Clustering : N Clusters
  double ECALClusterEtaRMS;//ECAL Clustering : Eta RMS
  double ECALClusterPhiRMS;//ECAL Clustering : Phi RMS
  double ECALClusterDRRMS;//ECAL Clustering : Delta R RMS 
  int ECALClusterDiscriminator;//ECAL Clustering :Discriminator



  //Constructor
  L2TauIsolationInfo()
    {
      ECALIsolConeCut=0.; 
      ECALIsolDiscriminator=0; 

      TowerIsolConeCut=0.;
      TowerIsolDiscriminator=0;

      ECALClusterNClusters=0;
      ECALClusterEtaRMS=0.;
      ECALClusterPhiRMS=0.;
      ECALClusterDRRMS=0.;
      ECALClusterDiscriminator=0;
      SeedTowerEt=100000.;

    }
  //Destructor
  ~L2TauIsolationInfo()
    {

    }


};

}
#endif

