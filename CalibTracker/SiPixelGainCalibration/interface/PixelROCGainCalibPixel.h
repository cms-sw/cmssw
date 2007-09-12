//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalibPixel

//---------------------------------------------------

#ifndef PixelROCGainCalibPixel_H
#define PixelROCGainCalibPixel_H

#include "TObject.h"
#include <vector>
#include <iostream>
#include "TGraph.h"
//#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"

class PixelROCGainCalibPixel
{
 private :
 
  std::vector<uint32_t> adcvalues;
  std::vector<uint32_t> nentries;
  std::vector<uint32_t> sumsquares;

  std::pair <double,double> pedanderror;
  std::pair <double,double> gainanderror;
 public :

  PixelROCGainCalibPixel(uint32_t npoints=60); 

  virtual ~PixelROCGainCalibPixel();

 //-- Setter/Getter
  uint32_t npoints(){return adcvalues.size();}
  void addPoint(uint32_t icalpoint, uint32_t adcval);
  TGraph *createGraph(TString thename, TString thetitle, const int ntimes,std::vector<uint32_t> vcalvalues);
  void init(uint32_t nvcal);
  void clearAllPoints();
  double getpoint(uint32_t icalpoint, uint32_t ntimes);
  double geterror(uint32_t icalpoint, uint32_t ntimes);
  double geterroreff(uint32_t icalpoint, uint32_t ntimes);
  double getefficiency(uint32_t icalpoint, uint32_t ntimes);
  void doAnalyticalFit(std::vector<uint32_t> vcalvalues, uint32_t vcallow, uint32_t vcalhigh); // does analytical fit
  uint32_t getsquaresum(uint32_t icalpoint) {return sumsquares[icalpoint];}
  uint32_t getsum(uint32_t icalpoint) {return adcvalues[icalpoint];}
  uint32_t getentries(uint32_t icalpoint){return nentries[icalpoint];}
  double getpedestal(){return pedanderror.first;}
  double getgain(){return gainanderror.first;}
  double getpedestalerror() {return pedanderror.second;}
  double getgainerror(){return gainanderror.second;}
  bool isfilled();

};

#endif

