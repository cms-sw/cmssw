/* 
Functions to return total energy contained in NxN (3x3/5x5/7x7)
Hcal towers aroud a given DetId. 

Inputs : 
1. HcalTopology, 
2. DetId around which NxN is to be formed, 
3. HcalRecHitCollection,
4. Number of towers to be navigated along eta and phi along 
   one direction (navigation is done alone +-deta and +-dphi).
5. option to include HO

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/

#ifndef CalibrationIsolatedParticleseHCALMatrixExtra_h
#define CalibrationIsolatedParticleseHCALMatrixExtra_h

// system include files
#include <memory>
#include <map>
#include <sstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"

namespace spr {

  template <typename T>
  std::vector<std::pair<DetId, double> > eHCALmatrixCell(const HcalTopology* topology,
                                                         const DetId& det,
                                                         edm::Handle<T>& hits,
                                                         int ieta,
                                                         int iphi,
                                                         bool includeHO = false,
                                                         double hbThr = -100,
                                                         double heThr = -100,
                                                         double hfThr = -100,
                                                         double hoThr = -100,
                                                         bool debug = false);

  template <typename T>
  std::pair<double, int> eHCALmatrixTotal(const HcalTopology* topology,
                                          const DetId& det,
                                          edm::Handle<T>& hits,
                                          int ieta,
                                          int iphi,
                                          bool includeHO = false,
                                          double hbThr = -100,
                                          double heThr = -100,
                                          double hfThr = -100,
                                          double hoThr = -100,
                                          bool debug = false);

  template <typename T>
  double energyHCALmatrix(const HcalTopology* topology,
                          const DetId& det,
                          edm::Handle<T>& hits,
                          int ieta,
                          int iphi,
                          bool includeHO = false,
                          double hbThr = -100,
                          double heThr = -100,
                          double hfThr = -100,
                          double hoThr = -100,
                          bool debug = false);

  template <typename T>
  double energyHCAL(std::vector<DetId>& vNeighboursDetId,
                    std::vector<DetId>& dets,
                    const HcalTopology* topology,
                    edm::Handle<T>& hits,
                    bool includeHO = false,
                    double hbThr = -100,
                    double heThr = -100,
                    double hfThr = -100,
                    double hoThr = -100,
                    bool debug = false);

  template <typename T>
  std::vector<std::pair<DetId, double> > energyDetIdHCAL(std::vector<DetId>& vdets,
                                                         edm::Handle<T>& hits,
                                                         double hbThr = -100,
                                                         double heThr = -100,
                                                         double hfThr = -100,
                                                         double hoThr = -100,
                                                         bool debug = false);

}  // namespace spr

#include "Calibration/IsolatedParticles/interface/eHCALMatrixExtra.icc"
#endif
