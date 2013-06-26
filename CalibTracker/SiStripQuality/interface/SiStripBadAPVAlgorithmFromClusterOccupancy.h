// -*- C++ -*-
//
// Package:    SiStripQuality
// Class:      SiStripBadAPVAlgorithmFromClusterOccupancy
// 
/**\class SiStripBadAPVAlgorithmFromClusterOccupancy SiStripBadAPVAlgorithmFromClusterOccupancy.h CalibTracker/SiStripQuality/src/SiStripBadAPVAlgorithmFromClusterOccupancy.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gordon KAUSSEN
//         Created:  Wed Jan 28 09:11:10 CEST 2009
// $Id: SiStripBadAPVAlgorithmFromClusterOccupancy.h,v 1.7 2013/01/11 04:57:47 wmtan Exp $
//
//

#ifndef CalibTracker_SiStripQuality_SiStripBadAPVAlgorithmFromClusterOccupancy_H
#define CalibTracker_SiStripQuality_SiStripBadAPVAlgorithmFromClusterOccupancy_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>

#include "TMath.h"
#include "TTree.h"
#include "TFile.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

class SiStripQuality;
class TrackerTopology;

class SiStripBadAPVAlgorithmFromClusterOccupancy{

public:
  typedef SiStrip::QualityHistosMap HistoMap;  
  
  SiStripBadAPVAlgorithmFromClusterOccupancy(const edm::ParameterSet&, const TrackerTopology*);

  virtual ~SiStripBadAPVAlgorithmFromClusterOccupancy();

  void setLowOccupancyThreshold(long double low_occupancy){lowoccupancy_=low_occupancy;}
  void setHighOccupancyThreshold(long double high_occupancy){highoccupancy_=high_occupancy;}
  void setAbsoluteLowThreshold(long double absolute_low){absolutelow_=absolute_low;}
  void setNumberIterations(int number_iterations){numberiterations_=number_iterations;}
  void setAbsoluteOccupancyThreshold(long double occupancy){occupancy_=occupancy;}
  void setNumberOfEvents(double Nevents){Nevents_=Nevents;}
  void setMinNumOfEvents();
  void setOutputFileName(std::string OutputFileName, bool WriteOutputFile){OutFileName_=OutputFileName; WriteOutputFile_=WriteOutputFile;}
  void setTrackerGeometry(const TrackerGeometry* tkgeom){TkGeom = tkgeom;}
  void extractBadAPVs(SiStripQuality*,HistoMap&,edm::ESHandle<SiStripQuality>&);

 private:

  struct Apv{   

    uint32_t detrawId;
    int modulePosition;
    int numberApvs;
    double apvMedian[6];
    double apvabsoluteOccupancy[6];
  };

  void CalculateMeanAndRMS(std::vector<Apv>, std::pair<double,double>*, int);

  void AnalyzeOccupancy(SiStripQuality*, std::vector<Apv>&, std::pair<double,double>*, std::vector<unsigned int>&, edm::ESHandle<SiStripQuality>&);

  struct pHisto{   

    pHisto():_NEntries(0),_NBins(0){};
    TH1F* _th1f;
    int _NEntries;
    int _NBins;
  };

  long double lowoccupancy_;
  long double highoccupancy_;
  long double absolutelow_;
  int numberiterations_;
  double Nevents_;
  long double occupancy_;
  double minNevents_;
  std::string OutFileName_;
  bool WriteOutputFile_;
  bool UseInputDB_;
  const TrackerGeometry* TkGeom;
  const TrackerTopology* tTopo;

  SiStripQuality *pQuality;

  double stripOccupancy[6][128];
  double stripWeight[6][128];

  std::vector<Apv> medianValues_TIB_Layer1; std::pair<double,double> MeanAndRms_TIB_Layer1[7];
  std::vector<Apv> medianValues_TIB_Layer2; std::pair<double,double> MeanAndRms_TIB_Layer2[7];
  std::vector<Apv> medianValues_TIB_Layer3; std::pair<double,double> MeanAndRms_TIB_Layer3[7];
  std::vector<Apv> medianValues_TIB_Layer4; std::pair<double,double> MeanAndRms_TIB_Layer4[7];

  std::vector<Apv> medianValues_TOB_Layer1; std::pair<double,double> MeanAndRms_TOB_Layer1[7];
  std::vector<Apv> medianValues_TOB_Layer2; std::pair<double,double> MeanAndRms_TOB_Layer2[7];
  std::vector<Apv> medianValues_TOB_Layer3; std::pair<double,double> MeanAndRms_TOB_Layer3[7];
  std::vector<Apv> medianValues_TOB_Layer4; std::pair<double,double> MeanAndRms_TOB_Layer4[7];
  std::vector<Apv> medianValues_TOB_Layer5; std::pair<double,double> MeanAndRms_TOB_Layer5[7];
  std::vector<Apv> medianValues_TOB_Layer6; std::pair<double,double> MeanAndRms_TOB_Layer6[7];

  std::vector<Apv> medianValues_TIDPlus_Disc1; std::pair<double,double> MeanAndRms_TIDPlus_Disc1[7];
  std::vector<Apv> medianValues_TIDPlus_Disc2; std::pair<double,double> MeanAndRms_TIDPlus_Disc2[7];
  std::vector<Apv> medianValues_TIDPlus_Disc3; std::pair<double,double> MeanAndRms_TIDPlus_Disc3[7];

  std::vector<Apv> medianValues_TIDMinus_Disc1; std::pair<double,double> MeanAndRms_TIDMinus_Disc1[7];
  std::vector<Apv> medianValues_TIDMinus_Disc2; std::pair<double,double> MeanAndRms_TIDMinus_Disc2[7];
  std::vector<Apv> medianValues_TIDMinus_Disc3; std::pair<double,double> MeanAndRms_TIDMinus_Disc3[7];

  std::vector<Apv> medianValues_TECPlus_Disc1; std::pair<double,double> MeanAndRms_TECPlus_Disc1[7];
  std::vector<Apv> medianValues_TECPlus_Disc2; std::pair<double,double> MeanAndRms_TECPlus_Disc2[7];
  std::vector<Apv> medianValues_TECPlus_Disc3; std::pair<double,double> MeanAndRms_TECPlus_Disc3[7];
  std::vector<Apv> medianValues_TECPlus_Disc4; std::pair<double,double> MeanAndRms_TECPlus_Disc4[7];
  std::vector<Apv> medianValues_TECPlus_Disc5; std::pair<double,double> MeanAndRms_TECPlus_Disc5[7];
  std::vector<Apv> medianValues_TECPlus_Disc6; std::pair<double,double> MeanAndRms_TECPlus_Disc6[7];
  std::vector<Apv> medianValues_TECPlus_Disc7; std::pair<double,double> MeanAndRms_TECPlus_Disc7[7];
  std::vector<Apv> medianValues_TECPlus_Disc8; std::pair<double,double> MeanAndRms_TECPlus_Disc8[7];
  std::vector<Apv> medianValues_TECPlus_Disc9; std::pair<double,double> MeanAndRms_TECPlus_Disc9[7];

  std::vector<Apv> medianValues_TECMinus_Disc1; std::pair<double,double> MeanAndRms_TECMinus_Disc1[7];
  std::vector<Apv> medianValues_TECMinus_Disc2; std::pair<double,double> MeanAndRms_TECMinus_Disc2[7];
  std::vector<Apv> medianValues_TECMinus_Disc3; std::pair<double,double> MeanAndRms_TECMinus_Disc3[7];
  std::vector<Apv> medianValues_TECMinus_Disc4; std::pair<double,double> MeanAndRms_TECMinus_Disc4[7];
  std::vector<Apv> medianValues_TECMinus_Disc5; std::pair<double,double> MeanAndRms_TECMinus_Disc5[7];
  std::vector<Apv> medianValues_TECMinus_Disc6; std::pair<double,double> MeanAndRms_TECMinus_Disc6[7];
  std::vector<Apv> medianValues_TECMinus_Disc7; std::pair<double,double> MeanAndRms_TECMinus_Disc7[7];
  std::vector<Apv> medianValues_TECMinus_Disc8; std::pair<double,double> MeanAndRms_TECMinus_Disc8[7];
  std::vector<Apv> medianValues_TECMinus_Disc9; std::pair<double,double> MeanAndRms_TECMinus_Disc9[7];


  TFile* f;
  TTree* apvtree;

  uint32_t detrawid;
  int subdetid;
  int layer_ring;
  int disc;
  int isback;
  int isexternalstring;
  int iszminusside;
  int rodstringpetal;
  int isstereo;
  int module_number;
  int number_strips;
  int number_apvs;
  int apv_number;

  float global_position_x;
  float global_position_y;
  float global_position_z;

  double apvAbsoluteOccupancy;
  double apvMedianOccupancy;

  std::stringstream ss;   
};
#endif

