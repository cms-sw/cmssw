/* Class that provides the simple ECAL Clustering Algorithm
   for L2 Tau Trigger */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include <vector>
#include "DataFormats/TauReco/interface/L2TauIsolationInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#ifndef L2TAU_SIMPLECLUSTERING_H
#define L2TAU_SIMPLECLUSTERING_H



class L2TauSimpleClustering
{
 public:
  //Constructor 
  L2TauSimpleClustering();
  L2TauSimpleClustering(double);
    
  //Destructor
  ~L2TauSimpleClustering();

  //METHODS
  math::PtEtaPhiELorentzVectorCollection  clusterize(const math::PtEtaPhiELorentzVectorCollection&); //Do Clustering


 private:
  //VARIABLES
  double m_clusterRadius;     //Cluster Radius

};

#endif
