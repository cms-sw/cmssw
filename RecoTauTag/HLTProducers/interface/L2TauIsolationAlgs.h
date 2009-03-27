/*
L2 Tau trigger Isolation algorithms

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/


#ifndef L2TAUISOLATIONALGS_H
#define L2TAUISOLATIONALGS_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include <vector>
#include "DataFormats/TauReco/interface/L2TauIsolationInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace reco;

///////////////////---CLASS DEFINITION---////////////////////
//////////////////    L2TauECALCluster   ///////////////////



class L2TauECALCluster
{
 public:
  //Constructors
  L2TauECALCluster();
  L2TauECALCluster(const math::PtEtaPhiELorentzVector&); 
  ~L2TauECALCluster();


  //Return Functions
  double etac() const; //Eta of Center
  double phic() const; //Phi of Center
  double et() const; //Et (total)
  int nCrystals() const; //N Crystals

  //Add Crystal to Cluster
  void addCrystal(const math::PtEtaPhiELorentzVector&);//Add a Crystal


 private:
  //Coordinates of the center
  double m_etac;
  double m_phic;


  //Cluster et(total)
  double m_et;

  //Number of crystals
  int m_ncrystals;

};

typedef std::vector<L2TauECALCluster> L2TauECALClusterCollection;
typedef std::vector<L2TauECALCluster>::const_iterator L2TauECALClusterIt;



///////////////////---CLASS DEFINITION---////////////////////
//////////////////    L2TauECALClustering///////////////////


class L2TauECALClustering
{
 public:
  //Constructor 
  L2TauECALClustering();
  L2TauECALClustering(double);
    
  //Destructor
  ~L2TauECALClustering();


  //METHODS
  void run(const math::PtEtaPhiELorentzVectorCollection&,const CaloJet&,L2TauIsolationInfo&);


  
 private:
  //VARIABLES
  double m_clusterRadius;     //Cluster Radius
  L2TauECALClusterCollection m_clusters;//Cluster Container



  //METHODS
  void clusterize(const math::PtEtaPhiELorentzVectorCollection&); //Do Clustering
  double deltaR(const math::PtEtaPhiELorentzVector&,const L2TauECALCluster&) const;//DeltaR (corr) between Crystal,Cluster
  int nClusters() const; //Number of Clusters
  std::vector<double> clusterSeperation(const CaloJet&) const; //Spreading of Clusters
  
  

    
  



};

///////////////////---CLASS DEFINITION---////////////////////
//////////////////    L2TauECALIsolation ///////////////////

class L2TauECALIsolation
{
 public:
  L2TauECALIsolation();
  L2TauECALIsolation(double,double);

  ~L2TauECALIsolation();

 
  void run(const math::PtEtaPhiELorentzVectorCollection&,const CaloJet& ,L2TauIsolationInfo&);


 private:
  //METHODS;
  double isolatedEt( const math::PtEtaPhiELorentzVectorCollection& ,const CaloJet&) const; //Return the Isolated Et


  //VARIABLES;
  double m_innerCone; //Inner Cone
  double m_outerCone; //Outer Cone

};


///////////////////---CLASS DEFINITION---////////////////////
//////////////////  L2TauTowerIsolation ///////////////////


class L2TauTowerIsolation
{
 public:
  L2TauTowerIsolation();
  L2TauTowerIsolation(double,double,double);

  ~L2TauTowerIsolation();


  void run(const CaloJet&,L2TauIsolationInfo&);


 private:
  //METHODS;
  double isolatedEt(const CaloJet&) const;
  double seedTowerEt(const CaloJet&) const; 

  //VARIABLES;
  double m_innerCone;
  double m_outerCone;
  double m_towerEtThreshold;




};

#endif
