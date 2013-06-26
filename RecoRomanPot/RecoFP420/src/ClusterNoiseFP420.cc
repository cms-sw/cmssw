///////////////////////////////////////////////////////////////////////////////
// File: ClusterNoiseFP420.cc
// Date: 12.2006
// Description: ClusterNoiseFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"
using namespace std;

ClusterNoiseFP420::ClusterNoiseFP420(){}
ClusterNoiseFP420::~ClusterNoiseFP420(){}



	      //   for case of access from DB
/*

const std::vector<ClusterNoiseFP420::ElectrodData> & ClusterNoiseFP420::getElectrodNoiseVector(const uint32_t & DetId) const {
  ElectrodNoiseMapIterator mapiter=m_noises.find(DetId);
  if (mapiter!=m_noises.end())
    return mapiter->second;
  return ElectrodNoiseVector();
};

*/
