#ifndef FP420ClusterMain_h
#define FP420ClusterMain_h
   
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
// #include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"
#include <iostream>
#include <vector>


// class ClusterNoiseFP420;
class ClusterProducerFP420;

class FP420ClusterMain 
{
 public:
  

 //   FP420ClusterMain(const edm::ParameterSet& conf, int dn, int sn, int pn, int rn, int dh, int sh, int ph, int rh);
    FP420ClusterMain(const edm::ParameterSet& conf, int , int , int , int , int , int , int , int );
  //  FP420ClusterMain();

  ~FP420ClusterMain();

  /// Runs the algorithm

//         void run(const DigiCollectionFP420 &input,
//       	   ClusterCollectionFP420 &soutput,
//       	   const std::vector<ClusterNoiseFP420>& noise 
//     	   );
    void run(edm::Handle<DigiCollectionFP420> &input,
       	   std::auto_ptr<ClusterCollectionFP420> &soutput);

 private:


  ClusterProducerFP420 *threeThresholdFP420_;

  ClusterProducerFP420 *threeThresholdHPS240_;

  std::string clusterModeFP420;
  std::string clusterModeHPS240;

  //std::vector<HDigiFP420> collector;
  edm::ParameterSet conf_;

  FP420NumberingScheme * theFP420NumberingScheme;


//  double pitchX;          // pitchX
//  double pitchY;          // pitchY
//  double pitch;          // pitch automatic

  float moduleThicknessX; // plate thicknessX 
  float moduleThicknessY; // plate thicknessY 
  float moduleThickness; // plate thickness 

  int numStripsX, numStripsXW;    // number of strips in the moduleX
  int numStripsY, numStripsYW;    // number of strips in the moduleY
  int numStrips;    // number of strips in the module

  float Thick300;

 // Number of Detectors/Stations/planes/sensors:
  int dn0, sn0, pn0, rn0;
  int dh0, sh0, ph0, rh0;

 // Type of planes:
   int xytype;

   int verbosity;

 //float sigma1;
 //float sigma2;

// FP420
  double ElectronPerADCFP420;
  double ChannelThresholdFP420;
  double SeedThresholdFP420;
  double ClusterThresholdFP420;
  int MaxVoidsInClusterFP420;	

  bool validClusterizerFP420;
  double ENCFP420;
  double BadElectrodeProbabilityFP420;
  bool UseNoiseBadElectrodeFlagFromDBFP420;
// HPS240
  double ElectronPerADCHPS240;
  double ChannelThresholdHPS240;
  double SeedThresholdHPS240;
  double ClusterThresholdHPS240;
  int MaxVoidsInClusterHPS240;	

  bool validClusterizerHPS240;
  double ENCHPS240;
  double BadElectrodeProbabilityHPS240;
  bool UseNoiseBadElectrodeFlagFromDBHPS240;


};

#endif
