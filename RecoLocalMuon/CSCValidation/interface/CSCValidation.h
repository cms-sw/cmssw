#ifndef RecoLocalMuon_CSCValidation_H
#define RecoLocalMuon_CSCValidation_H

/** \class CSCValidation
 *
 * Simple package to validate local CSC reconstruction:
 *    DIGIS
 *    recHits
 *    segments
 *
 * This program merely unpacks collections and fills
 * a few simple histograms.  The idea is to compare
 * the histograms for one offline release and another
 * and look for unexpected differences.
 *
 * Michael Schmitt, Northwestern University, July 2007
 */

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "TVector3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class PSimHit;
class TFile;
class CSCLayer;
class CSCDetId;

class CSCValidation : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCValidation();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


protected:

private: 

  // some useful functions
  float      fitX(HepMatrix sp, HepMatrix ep);
  HepMatrix  getCharge3x3(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip);
  float      getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip);
  void       doEfficiencies(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments);
  void       getEfficiency(float bin, float Norm, std::vector<float> &eff);
  void       histoEfficiency(TH1F *readHisto, TH1F *writeHisto);

  // counter
  int nEventsAnalyzed;

  // my histograms
  TH1F *hCalibGainsS;
  TH1F *hCalibXtalkSL;
  TH1F *hCalibXtalkSR;
  TH1F *hCalibXtalkIL;
  TH1F *hCalibXtalkIR;
  TH1F *hCalibPedsP;
  TH1F *hCalibPedsR;
  TH1F *hCalibNoise33;
  TH1F *hCalibNoise34;
  TH1F *hCalibNoise35;
  TH1F *hCalibNoise44;
  TH1F *hCalibNoise45;
  TH1F *hCalibNoise46;
  TH1F *hCalibNoise55;
  TH1F *hCalibNoise56;
  TH1F *hCalibNoise57;
  TH1F *hCalibNoise66;
  TH1F *hCalibNoise67;
  TH1F *hCalibNoise77;


  TH1F *hWireAll;
  TH1F *hWireTBinAll;
  TH1F *hWirenGroupsTotal;
  TH1F *hWireCodeBroad;
  TH1F *hWireCodeNarrow1;
  TH1F *hWireCodeNarrow2;
  TH1F *hWireCodeNarrow3;
  TH1F *hWireCodeNarrow4;
  TH1F *hWireLayer1;
  TH1F *hWireLayer2;
  TH1F *hWireLayer3;
  TH1F *hWireLayer4;
  TH1F *hWireWire1;
  TH1F *hWireWire2;
  TH1F *hWireWire3;
  TH1F *hWireWire4;

  TH1F *hStripAll;
  TH1F *hStripADCAll;
  TH1F *hStripNFired;
  TH1F *hStripCodeBroad;
  TH1F *hStripCodeNarrow1;
  TH1F *hStripCodeNarrow2;
  TH1F *hStripCodeNarrow3;
  TH1F *hStripCodeNarrow4;
  TH1F *hStripLayer1;
  TH1F *hStripLayer2;
  TH1F *hStripLayer3;
  TH1F *hStripLayer4;
  TH1F *hStripStrip1;
  TH1F *hStripStrip2;
  TH1F *hStripStrip3;
  TH1F *hStripStrip4;

  TH1F *hRHCodeBroad;
  TH1F *hRHCodeNarrow1;
  TH1F *hRHCodeNarrow2;
  TH1F *hRHCodeNarrow3;
  TH1F *hRHCodeNarrow4;
  TH1F *hRHLayer1;
  TH1F *hRHLayer2;
  TH1F *hRHLayer3;
  TH1F *hRHLayer4;
  TH1F *hRHX1;
  TH1F *hRHX2;
  TH1F *hRHX3;
  TH1F *hRHX4;
  TH1F *hRHY1;
  TH1F *hRHY2;
  TH1F *hRHY3;
  TH1F *hRHY4;
  TH2F *hRHGlobal1;
  TH2F *hRHGlobal2;
  TH2F *hRHGlobal3;
  TH2F *hRHGlobal4;
  TH1F *hRHEff;
  TH1F *hRHResid11b;
  TH1F *hRHResid12;
  TH1F *hRHResid13;
  TH1F *hRHResid11a;
  TH1F *hRHResid21;
  TH1F *hRHResid22;
  TH1F *hRHResid31;
  TH1F *hRHResid32;
  TH1F *hRHResid41;
  TH1F *hRHResid42;
  TH1F *hSResid11b;
  TH1F *hSResid12;
  TH1F *hSResid13;
  TH1F *hSResid11a;
  TH1F *hSResid21;
  TH1F *hSResid22;
  TH1F *hSResid31;
  TH1F *hSResid32;
  TH1F *hSResid41;
  TH1F *hSResid42;
  TH1F *hRHSumQ11b;
  TH1F *hRHSumQ12;
  TH1F *hRHSumQ13;
  TH1F *hRHSumQ11a;
  TH1F *hRHSumQ21;
  TH1F *hRHSumQ22;
  TH1F *hRHSumQ31;
  TH1F *hRHSumQ32;
  TH1F *hRHSumQ41;
  TH1F *hRHSumQ42;
  TH1F *hRHRatioQ11b;
  TH1F *hRHRatioQ12;
  TH1F *hRHRatioQ13;
  TH1F *hRHRatioQ11a;
  TH1F *hRHRatioQ21;
  TH1F *hRHRatioQ22;
  TH1F *hRHRatioQ31;
  TH1F *hRHRatioQ32;
  TH1F *hRHRatioQ41;
  TH1F *hRHRatioQ42;
  TH1F *hRHTiming11a;
  TH1F *hRHTiming12;
  TH1F *hRHTiming13;
  TH1F *hRHTiming11b;
  TH1F *hRHTiming21;
  TH1F *hRHTiming22;
  TH1F *hRHTiming31;
  TH1F *hRHTiming32;
  TH1F *hRHTiming41;
  TH1F *hRHTiming42;


  TH1F *hSCodeBroad;
  TH1F *hSCodeNarrow1;
  TH1F *hSCodeNarrow2;
  TH1F *hSCodeNarrow3;
  TH1F *hSCodeNarrow4;
  TH1F *hSnHits1;
  TH1F *hSnHits2;
  TH1F *hSnHits3;
  TH1F *hSnHits4;
  TH1F *hSTheta1;
  TH1F *hSTheta2;
  TH1F *hSTheta3;
  TH1F *hSTheta4;
  TH2F *hSGlobal1;
  TH2F *hSGlobal2;
  TH2F *hSGlobal3;
  TH2F *hSGlobal4;
  TH1F *hSnhits;
  TH1F *hSEff;
  TH1F *hSChiSqProb;
  TH1F *hSGlobalTheta;
  TH1F *hSGlobalPhi;
  TH1F *hSnSegments;

  // tmp histos for Efficiency
  TH1F *hSSTE;
  TH1F *hRHSTE;

  //
  //
  // A struct for creating a Tree/Branch of position info
  struct posRecord {
    int endcap;
    int station;
    int ring;
    int chamber;
    int layer;
    float localx;
    float localy;
    float globalx;
    float globaly;
  } rHpos, segpos;

  //
  //
  // The root tree
  TTree *rHTree;
  TTree *segTree;

  //
  //
  // The root file for the histograms.
  TFile *theFile;

  // input parameters for this module
  // Root file name
  std::string rootFileName;
  // Flag for simulation
  bool isSimulation;

};
#endif
