#ifndef RecoEgamma_EgammaTools_HGCalShowerShapeHelper_h
#define RecoEgamma_EgammaTools_HGCalShowerShapeHelper_h

// system include files
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// external include files
#include <CLHEP/Vector/LorentzVector.h>
#include <Math/Point3D.h>
#include <Math/Point3Dfwd.h>
#include <TMatrixD.h>
#include <TVectorD.h>

// CMSSW include files
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/FWLite/interface/ESHandle.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class HGCalShowerShapeHelper {
  // Good to filter/compute/store this stuff beforehand as they are common to the shower shape variables.
  // No point in filtering, computing layer-wise centroids, etc. for each variable again and again.
  // Once intitialized, one can the calculate different variables one after another for a given object.
  // If a different set of preselections (E, ET, etc.) is required for a given object, then reinitialize using initPerObject(...).

  // In principle should consider the HGCalHSi and HGCalHSc hits (leakage) also.
  // Can have subdetector dependent thresholds and layer selection.
  // To be implemented.

public:
  static const double kLDWaferCellSize_;
  static const double kHDWaferCellSize_;

  void setLayerWiseInfo();

  struct ShowerWidths {
    double sigma2xx;
    double sigma2yy;
    double sigma2zz;

    double sigma2xy;
    double sigma2yz;
    double sigma2zx;

    double sigma2uu;
    double sigma2vv;
    double sigma2ww;

    ShowerWidths()
        : sigma2xx(0.0),
          sigma2yy(0.0),
          sigma2zz(0.0),
          sigma2xy(0.0),
          sigma2yz(0.0),
          sigma2zx(0.0),
          sigma2uu(0.0),
          sigma2vv(0.0),
          sigma2ww(0.0) {}
  };

  HGCalShowerShapeHelper(edm::ConsumesCollector &&sumes);

  void initPerEvent(const edm::EventSetup &iSetup, const std::vector<reco::PFRecHit> &recHits);

  void initPerObject(const std::vector<std::pair<DetId, float> > &hitsAndFracs,
                     double minHitE = 0,
                     double minHitET = 0,
                     int minLayer = 1,
                     int maxLayer = -1,
                     DetId::Detector subDet = DetId::HGCalEE);

  const double getCellSize(DetId detId);

  // Compute Rvar in a cylinder around the layer centroids
  const double getRvar(double cylinderR, double energyNorm, bool useFractions = true, bool useCellSize = true);

  // Compute PCA widths around the layer centroids
  const ShowerWidths getPCAWidths(double cylinderR, bool useFractions = false);

private:
  void setPFRecHitPtrMap(const std::vector<reco::PFRecHit> &recHits);
  void setFilteredHitsAndFractions(const std::vector<std::pair<DetId, float> > &hitsAndFracs);

  double minHitE_;
  double minHitET_;
  double minHitET2_;
  int minLayer_;
  int maxLayer_;
  int nLayer_;
  DetId::Detector subDet_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  hgcal::RecHitTools recHitTools_;

  std::unordered_map<uint32_t, const reco::PFRecHit *> pfRecHitPtrMap_;
  std::vector<std::pair<DetId, float> > hitsAndFracs_;
  std::vector<double> hitEnergies_;
  std::vector<double> hitEnergiesWithFracs_;

  ROOT::Math::XYZVector centroid_;
  std::vector<double> layerEnergies_;
  std::vector<ROOT::Math::XYZVector> layerCentroids_;
};

#endif
