#ifndef RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_
#define RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include <map>
#include <string>

class HiEgammaSCEnergyCorrectionAlgo {
public:
  // the Verbosity levels
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 };

  // public member functions
  HiEgammaSCEnergyCorrectionAlgo(float noise, const edm::ParameterSet &pSet, VerbosityLevel verbosity = pERROR);

  // take a SuperCluster and return a corrected SuperCluster
  reco::SuperCluster applyCorrection(const reco::SuperCluster &cl,
                                     const EcalRecHitCollection &rhc,
                                     reco::CaloCluster::AlgoId algoId,
                                     const CaloSubdetectorGeometry &geometry,
                                     const CaloTopology &topology) const;

private:
  // correction factor as a function of number of crystals,
  // BasicCluster algo and location in the detector
  float fNCrystals(int nCry, reco::CaloCluster::AlgoId algoId, EcalSubdetector theBase) const;
  float fBrem(float widthRatio, reco::CaloCluster::AlgoId algoId, EcalSubdetector theBase) const;
  float fEta(float eta, reco::CaloCluster::AlgoId algoId, EcalSubdetector theBase) const;
  float fEtEta(float et, float eta, reco::CaloCluster::AlgoId algoId, EcalSubdetector theBase) const;

  // Return the number of crystals in a BasicCluster above
  // 2sigma noise level
  int nCrystalsGT2Sigma(reco::BasicCluster const &seed, EcalRecHitCollection const &rhc) const;

  const float sigmaElectronicNoise_;

  //  the verbosity level
  const VerbosityLevel verbosity_;

  // parameters
  const std::vector<double> p_fEta_;
  const std::vector<double> p_fBremTh_, p_fBrem_;
  const std::vector<double> p_fEtEta_;

  const double minR9Barrel_;
  const double minR9Endcap_;
  const double maxR9_;
};

#endif /*RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_*/
