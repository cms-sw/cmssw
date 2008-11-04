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
// $Id: SiStripHotStripAlgorithmFromClusterOccupancy.h,v 1.2 2007/11/02 21:06:38 giordano Exp $
//
//

#ifndef CalibTracker_SiStripQuality_SiStripHotStripAlgorithmFromClusterOccupancy_H
#define CalibTracker_SiStripQuality_SiStripHotStripAlgorithmFromClusterOccupancy_H

// system include files
#include <memory>
#include <vector>
#include <sstream>

#include "TMath.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

class SiStripQuality;

class SiStripHotStripAlgorithmFromClusterOccupancy{

public:
  typedef SiStrip::QualityHistosMap HistoMap;  
  

  SiStripHotStripAlgorithmFromClusterOccupancy():prob_(1.E-7),MinNumEntries_(0),MinNumEntriesPerStrip_(0){};
  virtual ~SiStripHotStripAlgorithmFromClusterOccupancy();

  void setProbabilityThreshold(long double prob){prob_=prob;}
  void setMinNumEntries(unsigned short m){MinNumEntries_=m;}
  void setMinNumEntriesPerStrip(unsigned short m){MinNumEntriesPerStrip_=m;}
  void extractBadStrips(SiStripQuality*,HistoMap&);
  std::vector<uint32_t> getStripOccupancy(){return _StripOccupancy;}
  
 private:

  struct pHisto{   

    pHisto():_NEntries(0),_NEmptyBins(0){};
    TH1F* _th1f;
    int _NEntries;
    int _NEmptyBins;
  };

  void iterativeSearch(pHisto&,std::vector<unsigned int>&);
  void evaluatePoissonian(std::vector<long double>& , float& meanVal);

  long double prob_;
  unsigned short MinNumEntries_;
  unsigned short MinNumEntriesPerStrip_;

  SiStripQuality *pQuality;
  std::vector<uint32_t> _StripOccupancy;
  std::stringstream ss;   
};
#endif

