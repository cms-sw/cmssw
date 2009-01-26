// -*- C++ -*-
//
// Package:    SiStripQuality
// Class:      SiStripHotStripAlgorithmFromClusterOccupancy
// 
/**\class SiStripHotStripAlgorithmFromClusterOccupancy SiStripHotStripAlgorithmFromClusterOccupancy.h CalibTracker/SiStripQuality/src/SiStripHotStripAlgorithmFromClusterOccupancy.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
// $Id: SiStripHotStripAlgorithmFromClusterOccupancy.h,v 1.5 2008/11/24 14:55:43 kaussen Exp $
//
//

#ifndef CalibTracker_SiStripQuality_SiStripHotStripAlgorithmFromClusterOccupancy_H
#define CalibTracker_SiStripQuality_SiStripHotStripAlgorithmFromClusterOccupancy_H

// system include files
#include <memory>
#include <vector>
#include <sstream>
#include <iostream>

#include "TMath.h"
#include "TTree.h"
#include "TFile.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

class SiStripQuality;

class SiStripHotStripAlgorithmFromClusterOccupancy{

public:
  typedef SiStrip::QualityHistosMap HistoMap;  
  

  SiStripHotStripAlgorithmFromClusterOccupancy():prob_(1.E-7),MinNumEntries_(0),MinNumEntriesPerStrip_(0),Nevents_(0),occupancy_(0),OutFileName_("Occupancy.root")
  {minNevents_=Nevents_*occupancy_;}
  virtual ~SiStripHotStripAlgorithmFromClusterOccupancy();

  void setProbabilityThreshold(long double prob){prob_=prob;}
  void setMinNumEntries(unsigned short m){MinNumEntries_=m;}
  void setMinNumEntriesPerStrip(unsigned short m){MinNumEntriesPerStrip_=m;}
  void setOccupancyThreshold(long double occupancy){occupancy_=occupancy;minNevents_=occupancy_*Nevents_;}
  void setNumberOfEvents(uint32_t Nevents);
  void setOutputFileName(std::string OutputFileName, bool WriteOutputFile){OutFileName_=OutputFileName; WriteOutputFile_=WriteOutputFile;}
  void setTrackerGeometry(const TrackerGeometry* tkgeom){TkGeom = tkgeom;}
  void extractBadStrips(SiStripQuality*,HistoMap&);

 private:

  struct pHisto{   

    pHisto():_NEntries(0),_NEmptyBins(0),_SubdetId(0){};
    TH1F* _th1f;
    int _NEntries;
    int _NEmptyBins;
    int _SubdetId;
  };

  void iterativeSearch(pHisto&,std::vector<unsigned int>&,int);
  void evaluatePoissonian(std::vector<long double>& , long double& meanVal);

  long double prob_;
  unsigned short MinNumEntries_;
  unsigned short MinNumEntriesPerStrip_;
  uint32_t Nevents_;
  double minNevents_;
  long double occupancy_;
  std::string OutFileName_;
  bool WriteOutputFile_;
  const TrackerGeometry* TkGeom;

  SiStripQuality *pQuality;

  TFile* f;
  TTree* striptree;

  int detrawid;
  int subdetid;
  int layer_ring;
  int disc;
  int isback;
  int isexternalstring;
  int iszminusside;
  int rodstringpetal;
  int isstereo;
  int module_position;
  int number_strips;
  int strip_number;
  int apv_channel;

  float global_position_x;
  float global_position_y;
  float global_position_z;

  int isHot;
  int hotStripsPerAPV;
  int hotStripsPerModule;
  double stripOccupancy;
  int stripHits;
  double poissonProb;

  int ishot[768];
  int hotstripsperapv[6];
  int hotstripspermodule;
  double stripoccupancy[768];
  int striphits[768];
  double poissonprob[768];


  std::stringstream ss;   
};
#endif

