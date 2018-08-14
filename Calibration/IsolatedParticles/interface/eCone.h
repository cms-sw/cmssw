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

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

namespace spr{

  // Basic cone energy cluster for hcal simhits and hcal rechits
  template <typename T>
    double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, int detOnly=-1, int useRaw=0, bool debug=false);

  // Cone energy cluster for hcal simhits and hcal rechits
  // that returns vector of rechit IDs and hottest cell info
  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, std::vector<DetId>& coneRecHitDetIds, double& distFromHotCell, int& ietaHotCell, int& iphiHotCell, GlobalPoint& gposHotCell, int detOnly=-1, int useRaw=0, bool debug=false);

  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, std::vector<DetId>& coneRecHitDetIds, std::vector<double>& eHit, int useRaw=0, bool debug=false);
 
  // Cone energy cluster for hcal simhits and hcal rechits
  // that returns vector of rechit IDs and hottest cell info
  // AND info for making "hit maps"
  template <typename T>
  double eCone_hcal(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene, std::vector<DetId>& coneRecHitDetIds, double& distFromHotCell, int& ietaHotCell, int& iphiHotCell, GlobalPoint& gposHotCell, int detOnly=-1, int useRaw=0, bool debug=false);

  // Basic cone energy clustering for Ecal
  template <typename T>
  double eCone_ecal(const CaloGeometry* geo, edm::Handle<T>& barrelhits, edm::Handle<T>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, int& nRecHits, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  double eCone_ecal(const CaloGeometry* geo, edm::Handle<T>& barrelhits, edm::Handle<T>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, std::vector<DetId>& coneRecHitDetIds, std::vector<double>& eHit, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);

}

#include "eCone.icc"

#endif
