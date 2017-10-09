/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 * September 2016
 *
 **********************************************************************/
#ifndef RecoCTPPS_PixelLocal_RPixDetClusterizer_H
#define RecoCTPPS_PixelLocal_RPixDetClusterizer_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "RecoCTPPS/PixelLocal/interface/CTPPSPixelGainCalibrationDBService.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"

#include <vector>
#include <set>

class RPixCalibDigi : public CTPPSPixelDigi {

public:

RPixCalibDigi(unsigned char row, unsigned char col, unsigned short adc, unsigned short ele) : CTPPSPixelDigi(row,col,adc){
    electrons_ = ele;
  }

RPixCalibDigi() : CTPPSPixelDigi(){
    electrons_ = 0;
}

  int electrons() const {
    return electrons_;
  }
  void set_electrons(int a) {
    electrons_=a;
  }

private:

  int electrons_;

};


class RPixDetClusterizer{

public:

  RPixDetClusterizer(edm::ParameterSet const& conf);

  void buildClusters(unsigned int detId, const std::vector<CTPPSPixelDigi> &digi, std::vector<CTPPSPixelCluster> &clusters, const CTPPSPixelGainCalibrations * pcalibration, const CTPPSPixelAnalysisMask*  mask);
  int calibrate(unsigned int, int, int, int ,const CTPPSPixelGainCalibrations * pcalibration);

  void make_cluster( RPixCalibDigi const &aSeed,  std::vector<CTPPSPixelCluster> &clusters );
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



class RPixTempCluster{

public:

  RPixTempCluster()
  {
    isize=0; 
    curr=0; 
  }
  ~RPixTempCluster(){}

  static constexpr unsigned short MAXSIZE = 256;
  unsigned short adc[MAXSIZE];
  uint8_t row[MAXSIZE];
  uint8_t col[MAXSIZE];
  unsigned short isize;
  unsigned short curr;

  // stack interface (unsafe ok for use below)
  unsigned short top() const { return curr;}
  void pop() { ++curr;}   
  bool empty() { return curr==isize;}

  bool addPixel(unsigned char myrow, unsigned char mycol, unsigned short const iadc) {
    if (isize==MAXSIZE) return false;
    adc[isize]=iadc;
    row[isize]=myrow;
    col[isize++]=mycol;
    return true;
  }
 

};


#endif
