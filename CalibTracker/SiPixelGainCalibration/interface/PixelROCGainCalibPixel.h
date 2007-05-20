//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalibPixel

//---------------------------------------------------

#ifndef PixelROCGainCalibPixel_H
#define PixelROCGainCalibPixel_H

#include "TObject.h"
#include "TObjArray.h"
#include "TH1F.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"

class PixelROCGainCalibPixel : public TObject
{
 private :
 
  TObjArray *calibPoints;//("PixelROCGainCalibElement",20);//contains PixelROCGainElements

 public :

  PixelROCGainCalibPixel();

  virtual ~PixelROCGainCalibPixel(){calibPoints->Delete();}

 //- Accessible methods
  //- Object Status

 
  void Clear(Option_t* = "") {;}
  void Reset(Option_t* = "") {;}

  void Print(Option_t* = "") const {;}

 //-- Setter/Getter

  TObjArray  *getCalibPoints() const { return calibPoints;}

  void addPoint(unsigned int icalpoint, unsigned int adcval, unsigned int vcalval);
  void setVCalPoint(unsigned int icalpoint, unsigned int adcVcalval);
  TH1F *createHistogram(TString thename, TString thetitle, int nbins, float lowval, float highval);
  void init(unsigned int nvcal);

};

#endif

