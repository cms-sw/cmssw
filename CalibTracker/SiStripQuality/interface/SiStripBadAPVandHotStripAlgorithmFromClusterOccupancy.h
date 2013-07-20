// -*- C++ -*-
//
// Package:    SiStripQuality
// Class:      SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy
// 
/**\class SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.h CalibTracker/SiStripQuality/src/SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gordon KAUSSEN
//         Created:  Wed Jan 28 09:11:10 CEST 2009
// $Id: SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.h,v 1.3 2013/01/11 04:57:47 wmtan Exp $
//
//

#ifndef CalibTracker_SiStripQuality_SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy_H
#define CalibTracker_SiStripQuality_SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy_H

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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class SiStripQuality;
class TrackerTopology;

class SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy{

public:
  typedef SiStrip::QualityHistosMap HistoMap;  
  
  SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy(const edm::ParameterSet&, const TrackerTopology*);

  virtual ~SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy();

  void setProbabilityThreshold(long double prob){prob_=prob;}
  void setMinNumEntries(unsigned short m){MinNumEntries_=m;}
  void setMinNumEntriesPerStrip(unsigned short m){MinNumEntriesPerStrip_=m;}
  void setLowOccupancyThreshold(long double low_occupancy){lowoccupancy_=low_occupancy;}
  void setHighOccupancyThreshold(long double high_occupancy){highoccupancy_=high_occupancy;}
  void setAbsoluteLowThreshold(long double absolute_low){absolutelow_=absolute_low;}
  void setNumberIterations(int number_iterations){numberiterations_=number_iterations;}
  void setAbsoluteOccupancyThreshold(long double absolute_occupancy){absolute_occupancy_=absolute_occupancy;}
  void setNumberOfEvents(double Nevents){Nevents_=Nevents;}
  void setMinNumOfEvents();
  void setOutputFileName(std::string OutputFileName, bool WriteOutputFile, std::string DQMOutfileName, bool WriteDQMHistograms){OutFileName_=OutputFileName; WriteOutputFile_=WriteOutputFile; DQMOutfileName_=DQMOutfileName; WriteDQMHistograms_=WriteDQMHistograms;}
  void setTrackerGeometry(const TrackerGeometry* tkgeom){TkGeom = tkgeom;}
  void extractBadAPVSandStrips(SiStripQuality*,HistoMap&,edm::ESHandle<SiStripQuality>&);

 private:

  struct Apv{   

    uint32_t detrawId;
    int modulePosition;
    int numberApvs;
    double apvMedian[6];
    int apvabsoluteOccupancy[6];
    TH1F* th1f[6];
    int NEntries[6];
    int NEmptyBins[6];
  };

  void CalculateMeanAndRMS(std::vector<Apv>, std::pair<double,double>*, int);

  void AnalyzeOccupancy(SiStripQuality*, std::vector<Apv>&, std::pair<double,double>*, std::vector<unsigned int>&, edm::ESHandle<SiStripQuality>&);

  void iterativeSearch(Apv&,std::vector<unsigned int>&,int);

  void evaluatePoissonian(std::vector<long double>& , long double& meanVal);

  void setBasicTreeParameters(int detid);

  void initializeDQMHistograms();

  void fillStripDQMHistograms();

  long double prob_;
  unsigned short MinNumEntries_;
  unsigned short MinNumEntriesPerStrip_;
  long double lowoccupancy_;
  long double highoccupancy_;
  long double absolutelow_;
  int numberiterations_;
  double Nevents_;
  long double absolute_occupancy_;
  double minNevents_;
  std::string OutFileName_;
  bool WriteOutputFile_;
  std::string DQMOutfileName_;
  bool WriteDQMHistograms_;
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
  float strip_global_position_x;
  float strip_global_position_y;
  float strip_global_position_z;

  int    apvAbsoluteOccupancy;
  double apvMedianOccupancy;
  int    isBad;

  TTree* striptree;
  int strip_number;
  int apv_channel;

  int isHot;
  int hotStripsPerAPV;
  int hotStripsPerModule;
  double singleStripOccupancy;
  int stripHits;
  double poissonProb;

  int ishot[128];
  int hotstripsperapv[6];
  int hotstripspermodule;
  double stripoccupancy[128];
  int striphits[128];
  double poissonprob[128];

  std::stringstream ss;

  std::ostringstream oss;

  DQMStore* dqmStore;

  MonitorElement* tmp;
  TProfile* tmp_prof;

  // Histograms
  // indexes in these arrays are [SubDetId-2][LayerN]
  // histograms for [SubDetId-2][0] are global for the subdetector
  // histogram for [0][0] is global for the tracker
  
  TH2F* medianVsAbsoluteOccupancy[5][10];
  TH1F* medianOccupancy[5][10];
  TH1F* absoluteOccupancy[5][10];

  std::vector<TH2F*> distanceVsStripNumber;
  std::vector<TProfile*> pfxDistanceVsStripNumber;
  std::vector<TH1F*> projXDistanceVsStripNumber;
  std::vector<TH1F*> projYDistanceVsStripNumber;

  std::vector<TH2F*> occupancyVsStripNumber;
  std::vector<TProfile*> pfxOccupancyVsStripNumber;
  std::vector<TH1F*> projYOccupancyVsStripNumber;
  std::vector<TH2F*> occupancyHotStripsVsStripNumber;
  std::vector<TProfile*> pfxOccupancyHotStripsVsStripNumber;
  std::vector<TH1F*> projYOccupancyHotStripsVsStripNumber;
  std::vector<TH2F*> occupancyGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxOccupancyGoodStripsVsStripNumber;
  std::vector<TH1F*> projYOccupancyGoodStripsVsStripNumber;

  std::vector<TH2F*> poissonProbVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbVsStripNumber;
  std::vector<TH1F*> projYPoissonProbVsStripNumber;
  std::vector<TH2F*> poissonProbHotStripsVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbHotStripsVsStripNumber;
  std::vector<TH1F*> projYPoissonProbHotStripsVsStripNumber;
  std::vector<TH2F*> poissonProbGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbGoodStripsVsStripNumber;
  std::vector<TH1F*> projYPoissonProbGoodStripsVsStripNumber;
   
  std::vector<TH2F*> nHitsVsStripNumber;
  std::vector<TProfile*> pfxNHitsVsStripNumber;
  std::vector<TH1F*> projXNHitsVsStripNumber;
  std::vector<TH1F*> projYNHitsVsStripNumber;
  std::vector<TH2F*> nHitsHotStripsVsStripNumber;
  std::vector<TProfile*> pfxNHitsHotStripsVsStripNumber;
  std::vector<TH1F*> projXNHitsHotStripsVsStripNumber;
  std::vector<TH1F*> projYNHitsHotStripsVsStripNumber;
  std::vector<TH2F*> nHitsGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxNHitsGoodStripsVsStripNumber;
  std::vector<TH1F*> projXNHitsGoodStripsVsStripNumber;
  std::vector<TH1F*> projYNHitsGoodStripsVsStripNumber;

  std::vector<std::string> subDetName;
  std::vector<unsigned int> nLayers;
  std::vector<std::string> layerName;

  std::vector<unsigned int> vHotStripsInModule;
  unsigned int distance;
  unsigned int distanceR, distanceL;

  std::string outfilename;
};
#endif

