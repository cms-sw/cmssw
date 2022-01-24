#ifndef FP420ClusterMain_h
#define FP420ClusterMain_h

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterProducerFP420.h"
#include <iostream>
#include <vector>
#include <memory>

class ClusterNoiseFP420;

class FP420ClusterMain {
public:
  FP420ClusterMain(const edm::ParameterSet &conf, int dn, int sn, int pn, int rn);
  //  FP420ClusterMain();

  /// Runs the algorithm

  //         void run(const DigiCollectionFP420 &input,
  //       	   ClusterCollectionFP420 &soutput,
  //       	   const std::vector<ClusterNoiseFP420>& noise
  //     	   );
  void run(edm::Handle<DigiCollectionFP420> &input, ClusterCollectionFP420 *soutput) const;

private:
  std::unique_ptr<const ClusterProducerFP420> threeThreshold_;
  std::string clusterMode_;

  //std::vector<HDigiFP420> collector;

  bool validClusterizer_;
  double ElectronPerADC_;
  double ENC_;
  double BadElectrodeProbability_;
  bool UseNoiseBadElectrodeFlagFromDB_;

  double ChannelThreshold;
  double SeedThreshold;
  double ClusterThreshold;
  int MaxVoidsInCluster;

  double ldriftX;
  double ldriftY;
  double pitchX;                // pitchX
  double pitchY;                // pitchY
  float moduleThicknessX;       // plate thicknessX
  float moduleThicknessY;       // plate thicknessY
  int numStripsX, numStripsXW;  // number of strips in the moduleX
  int numStripsY, numStripsYW;  // number of strips in the moduleY

  float Thick300;

  // Number of Detectors:
  int dn0;
  // Number of Stations:
  int sn0;
  // Number of planes:
  int pn0;
  // Number of sensors:
  int rn0;
  // Type of planes:
  int xytype;

  int verbosity;

  //float sigma1_;
  //float sigma2_;
};

#endif
