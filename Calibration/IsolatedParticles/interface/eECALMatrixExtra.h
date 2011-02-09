// -*- C++ -*
/* 
The function eECALmatrix returns total energy contained in 
NxN crystal matrix for EcalRecHits or PCaloSimHits.

Inputs : 
1. CaloNavigator at the DetId around which NxN has to be formed
2. The EcalRecHitCollection  and 
3. Number of crystals to be navigated along eta and phi along 
   one direction (navigation is done alone +-deta and +-dphi).

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/


#ifndef CalibrationIsolatedParticleseECALMatrixExtra_h
#define CalibrationIsolatedParticleseECALMatrixExtra_h

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"

namespace spr{

  // Energy in NxN crystal matrix
  template< typename T>
  double eECALmatrix(CaloNavigator<DetId>& navigator,edm::Handle<T>& hits, int ieta, int iphi, bool debug=false);

  template< typename T>
  std::vector<std::pair<DetId,double> > eECALmatrixCell(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, double ebThr=-100, double eeThr=-100, bool debug=false);

  // Energy in ietaXiphi crystal matrix
  template< typename T>
  std::pair<double,int> eECALmatrixTotal(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, double ebThr=-100, double eeThr=-100, bool debug=false);
  
  // returns vector of hits in NxN matrix 
  template <typename T>
  std::vector<std::pair<DetId,double> > energyECALCell(std::vector<DetId>& vdets, edm::Handle<T>& hitsEB,  edm::Handle<T>& hitsEE, double ebThr=-100, double eeThr=-100, bool debug=false);
  
}

#include "Calibration/IsolatedParticles/interface/eECALMatrixExtra.icc"

#endif
