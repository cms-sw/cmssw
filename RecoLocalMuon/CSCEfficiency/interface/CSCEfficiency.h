#ifndef RecoLocalMuon_CSCEfficiency_H
#define RecoLocalMuon_CSCEfficiency_H

/** \class CSCEfficiency
 *
 * Efficiency calculations 
 * Stoyan Stoynev, Northwestern University
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "RecoLocalMuon/CSCRecHit/src/CSCWireCluster.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//IBL - test to read simhits a la digi/rechit validation
//#include "Validation/MuonCSCDigis/interface/PSimHitMap.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"

#include "Utilities/Timing/interface/TimerStack.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"
#include "TVector3.h"
#include "TProfile.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"

#include <iostream>
#include <iomanip>
#include <fstream>
//#include "Math/Interpolator.h"



//#include "myHistograms.h"

#include <vector>
#include <map>
#include <string>

#define SQR(x) ((x)*(x))

#define LastCh 36
#define FirstCh  1
#define NumCh (LastCh-FirstCh+1)
//---- Useful structure 
   struct ChamberRecHits{
    std::map<int,  std::vector <double> > RecHitsPosXlocal; // layers vs number of hits
    std::map<int,  std::vector <double> > RecHitsPosYlocal;
    std::map<int,  std::vector <double> > RecHitsPosX; 
    std::map<int,  std::vector <double> > RecHitsPosY;
    std::map<int,  std::vector <double> > RecHitsPosZ;
    std::map<int,  int > NRecHits;
    std::map<int,  int > TheRightRecHit;
    int nSegments;
    ChamberRecHits(){
      std::vector <double> Zero;
      nSegments = 0;
      for (int iLayer=0;iLayer<6;iLayer++){
	RecHitsPosXlocal[iLayer] = Zero;
	RecHitsPosYlocal[iLayer] = Zero;
	RecHitsPosX[iLayer] = Zero;
	RecHitsPosY[iLayer] = Zero;
	RecHitsPosZ[iLayer] = Zero;
	TheRightRecHit[iLayer] = -1;
	NRecHits[iLayer] = 0;
      }
    }
  } ; 
  struct SetOfRecHits{
    int nEndcap ;
    int nStation;
    int nRing ;
    int Nchamber;
    ChamberRecHits sChamber;
  } ;
//

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

//class PSimHit;
class TFile;
class CSCLayer;
class CSCDetId;

class CSCEfficiency : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCEfficiency(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCEfficiency();

  // Operations

  //---- Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

protected:

private:
  //---- Variables

  SetOfRecHits  (*all_RecHits)[2][4][3][ NumCh];

  //---- Strip number and ADCPeak
  std::vector <std::pair <int, float> > AllStrips[2][4][3][NumCh][6];//endcap/station/ring/chamber/layer

  //---- WG number and Y-position, time bin
  std::vector <std::pair <std::pair <int, float>, int> > AllWG[2][4][3][NumCh][6];//endcap/station/ring/chamber/layer

  //---- Functions
  void CalculateEfficiencies(const edm::Event & event, const edm::EventSetup& eventSetup,
			  std::vector<double> &Pos , std::vector<double> &Dir, int NSegFound);
  void RecHitEfficiency(double Yprime, double Yright, int iCh);
  bool LCT_Efficiencies(edm::Handle<CSCALCTDigiCollection> alcts,     
                                  edm::Handle<CSCCLCTDigiCollection> clcts,
                                  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts, int iCh, int cond);
  void StripWire_Efficiencies (int iCh);
  void Segment_Efficiency (int iCh, int NSegmentsFound);
  //---- counter
  int nEventsAnalyzed;

  //---- Histograms 
    TH1F * DataFlow;
    TH2F * XY_ALCTmissing;
    //
    TH1F * dydz_Eff_ALCT;
    TH1F * dydz_All_ALCT;
    TH1F * FINAL_dydz_Efficiency_ALCT;
    //
    TH1F * EfficientSegments;
    TH1F * AllSegments;
    TH1F * EfficientRechits_inSegment;
    TH1F * InefficientSingleHits;
    TH1F * AllSingleHits;
    TH2F * XvsY_InefficientRecHits;
    TH2F * XvsY_InefficientRecHits_good;
    TH2F * XvsY_InefficientSegments;
    TH2F * XvsY_InefficientSegments_good;
    TH1F * EfficientRechits;
    TH1F * EfficientRechits_good;
    TH1F * EfficientLCTs;
    TH1F * EfficientStrips;
    TH1F * EfficientWireGroups;
    TH1F * Rechit_eff;
    std::vector<TH2F *> XvsY_InefficientRecHits_inSegment;
    std::vector<TH1F *> Y_InefficientRecHits_inSegment;
    std::vector<TH1F *> Y_AllRecHits_inSegment; 
//---- Efficiencies
    TH1F * FINAL_Segment_Efficiency;
    TH1F * FINAL_Rechit_inSegment_Efficiency;
    TH1F * FINAL_Attachment_Efficiency;
    TH1F * FINAL_Rechit_Efficiency;
    TH1F * FINAL_Rechit_Efficiency_good;
    TH1F * FINAL_LCTs_Efficiency;
    TH1F * FINAL_Strip_Efficiency;
    TH1F * FINAL_WireGroup_Efficiency;
    std::vector<TH1F *> FINAL_Y_RecHit_InSegment_Efficiency;
//---- Histograms set (chambers)...
  struct ChamberHistos{
    TH1F * EfficientRechits_inSegment;
    TH1F * InefficientSingleHits;
    TH1F * AllSingleHits;
    TH2F * XvsY_InefficientRecHits;
    TH2F * XvsY_InefficientRecHits_good;
    TH2F * XvsY_InefficientSegments;
    TH2F * XvsY_InefficientSegments_good;
    TH1F * EfficientRechits;
    TH1F * EfficientRechits_good;
    TH1F * EfficientLCTs;
    TH1F * EfficientStrips;
    TH1F * EfficientWireGroups;
    std::vector<TH2F *> XvsY_InefficientRecHits_inSegment;
    std::vector<TH1F *> Y_InefficientRecHits_inSegment;
    std::vector<TH1F *> Y_AllRecHits_inSegment;
//---- Efficiencies
    TH1F * FINAL_Rechit_inSegment_Efficiency;
    TH1F * FINAL_Attachment_Efficiency;
    TH1F * FINAL_Rechit_Efficiency;
    TH1F * FINAL_Rechit_Efficiency_good;
    TH1F * FINAL_LCTs_Efficiency;
    TH1F * FINAL_Strip_Efficiency;
    TH1F * FINAL_WireGroup_Efficiency;
    std::vector<TH1F *> FINAL_Y_RecHit_InSegment_Efficiency;
    //Auto_ptr...
    //std::auto_ptr<TH1F> perLayerIneffRecHit;
    
  }ChHist[LastCh-FirstCh+1];
  // utility flag
  bool flag;
  // dy/dz of the segment in the ref. station  
  float seg_dydz;
  // printalot debug output
  bool printalot;

  // Data (true) or MC (false) input file
  bool DATA;

  
  // calculate efficiencies from existing file (if set to "true")
  bool update;

  //---- The root file for the histograms
  TFile *theFile;

  //---- input parameters for this module
  std::string mycscunpacker;
  //---- Root file name
  std::string rootFileName;
  int WorkInEndcap, ExtrapolateFromStation; 
  int ExtrapolateToStation, ExtrapolateToRing;
  //---- Utility functions
  bool GoodRegion(double Yreal, double Yborder, int Station, int Ring, int Chamber);
  bool GoodLocalRegion(double Xreal, double Yreal, int Station, int Ring, int Chamber);
  bool CheckLocal(double Yreal, double Yborder, int Station, int Ring, bool withinChamberOnly);
  void getEfficiency(float bin, float Norm, std::vector <float> &eff);
  void histoEfficiency(TH1F *readHisto, TH1F *writeHisto, int flag);
  const char*  ChangeTitle(const char * name);
  double Extrapolate1D(double initPosition, double initDirection, double ParameterOfTheLine);
  double LineParam(double z1Position, double z2Position, double z1Direction);
  //IBL - test to read simhits a la digi/rechit validation
  //PSimHitMap theSimHitMap;
};

#endif




