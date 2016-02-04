
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

BremRecoveryPhiRoadAlgo::BremRecoveryPhiRoadAlgo(const edm::ParameterSet& pset)
{

   // get barrel and endcap parametersets
   edm::ParameterSet barrelPset = pset.getParameter<edm::ParameterSet>("barrel");
   edm::ParameterSet endcapPset = pset.getParameter<edm::ParameterSet>("endcap");

   // set barrel parameters
   etVec_ = barrelPset.getParameter<std::vector<double> >("etVec");
   cryVec_ = barrelPset.getParameter<std::vector<int> >("cryVec");
   cryMin_ = barrelPset.getParameter<int>("cryMin");

   // set endcap parameters
   a_ = endcapPset.getParameter<double>("a");
   b_ = endcapPset.getParameter<double>("b");
   c_ = endcapPset.getParameter<double>("c");

}

int BremRecoveryPhiRoadAlgo::barrelPhiRoad(double et)
{

   // 
   // Take as input the ET in 5x5 crystals
   // and compute the optimal phi road 
   // as a number of crystals

   for (unsigned int i = 0; i < cryVec_.size(); ++i)
   {
      if (et < etVec_[i]) return cryVec_[i];
   }
   return cryMin_;

}

double BremRecoveryPhiRoadAlgo::endcapPhiRoad(double energy)
{

   //
   // Take as input the energy in the seed BasicCluster
   // and return the optimal phi road
   // length in radians

   return ((a_ / (energy + b_)) + c_);

}

