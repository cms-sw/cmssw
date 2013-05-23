#ifndef CalibrationIsolatedParticleseCone_h
#define CalibrationIsolatedParticleseCone_h

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

namespace spr{

  // Basic cone energy cluster for hcal simhits and hcal rechits
  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits);

  // Cone energy cluster for hcal simhits and hcal rechits
  // that returns vector of rechit IDs and hottest cell info
  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, std::vector<DetId>& coneRecHitDetIds, double& distFromHotCell, int& ietaHotCell, int& iphiHotCell, GlobalPoint& gposHotCell);

 
  // Cone energy cluster for hcal simhits and hcal rechits
  // that returns vector of rechit IDs and hottest cell info
  // AND info for making "hit maps"
  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene, std::vector<DetId>& coneRecHitDetIds, double& distFromHotCell, int& ietaHotCell, int& iphiHotCell, GlobalPoint& gposHotCell);

  // Basic cone energy clustering for Ecal
  template <typename T>
  double eCone_ecal(const CaloGeometry* geo, edm::Handle<T>& barrelhits, edm::Handle<T>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits);

}

#include "eCone.icc"

#endif
