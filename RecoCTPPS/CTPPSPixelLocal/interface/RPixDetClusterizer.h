/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 * September 2016
 *
 **********************************************************************/
#ifndef RecoCTPPS_CTPPSPixelCluster_DET_CLUSTERIZER_H
#define RecoCTPPS_CTPPSPixelCluster_DET_CLUSTERIZER_H

//#define _DAQ_

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#ifndef _DAQ_
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "RecoCTPPS/CTPPSPixelLocal/interface/CTPPSPixelGainCalibrationDBService.h"
#endif
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"

#ifdef _DAQ_
#include "CondTools/CTPPS/interface/CTPPSPixelDAQCalibration.h"
#endif

#include <vector>
#include <set>

class RPixCalibDigi : public CTPPSPixelDigi {

public:

RPixCalibDigi(unsigned char row, unsigned char col, unsigned short adc, unsigned short ele) : CTPPSPixelDigi(row,col,adc){
    electrons_ = ele;
  }

RPixCalibDigi() : CTPPSPixelDigi(){}

  int electrons() const {
    return electrons_;
  }
  void set_electrons(int a) {
    electrons_=a;
  }

  int electrons_;

};


class RPixDetClusterizer{

public:

  RPixDetClusterizer(edm::ParameterSet const& conf);
#ifdef _DAQ_
CTPPSPixelDAQCalibration * theDAQcalibration;
 void buildClusters(unsigned int detId, const std::vector<CTPPSPixelDigi> &digi, std::vector<CTPPSPixelCluster> &clusters, const CTPPSPixelAnalysisMask*  mask);
  int calibrate(unsigned int, int, int, int);
#else
  void buildClusters(unsigned int detId, const std::vector<CTPPSPixelDigi> &digi, std::vector<CTPPSPixelCluster> &clusters, const CTPPSPixelGainCalibrations * pcalibration, const CTPPSPixelAnalysisMask*  mask);
  int calibrate(unsigned int, int, int, int ,const CTPPSPixelGainCalibrations * pcalibration);
#endif

  void make_cluster( RPixCalibDigi aSeed,  std::vector<CTPPSPixelCluster> &clusters );
  ~RPixDetClusterizer();


private:

  std::set<CTPPSPixelDigi> rpix_digi_set_;
  std::set<RPixCalibDigi> calib_rpix_digi_set_;
  const edm::ParameterSet &params_;
  int verbosity_;
  unsigned short SeedADCThreshold_;
  unsigned short ADCThreshold_;
  double ElectronADCGain_;
  int VcaltoElectronGain_;
  int VcaltoElectronOffset_;
  bool doSingleCalibration_;
  std::string CalibrationFile_;
  std::vector<RPixCalibDigi> SeedVector_;
  
};



class tempCluster{

public:

  tempCluster()
  {
    isize=0; 
    curr=0; 
    rowmin=255; 
    colmin=255;
  }
  ~tempCluster(){}

  static constexpr unsigned short MAXSIZE = 256;
  unsigned short adc[MAXSIZE];
  unsigned char row[MAXSIZE];
  unsigned char col[MAXSIZE];
  unsigned char rowmin;
  unsigned char colmin;
  unsigned short isize;
  unsigned short curr;

  // stack interface (unsafe ok for use below)
  unsigned short top() const { return curr;}
  void pop() { ++curr;}   
  bool empty() { return curr==isize;}

  bool addPixel(unsigned char myrow, unsigned char mycol, unsigned short const iadc) {
    if (isize==MAXSIZE) return false;
    rowmin=std::min(rowmin,myrow);
    colmin=std::min(colmin,mycol);
    adc[isize]=iadc;
    row[isize]=myrow;
    col[isize++]=mycol;
    return true;
  }
 

};


#endif
