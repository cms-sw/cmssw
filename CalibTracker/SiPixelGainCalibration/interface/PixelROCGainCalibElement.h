//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalibElement

//---------------------------------------------------

#ifndef PixelROCGainCalibElement_H
#define PixelROCGainCalibElement_H

#include "TObject.h"
#include "TROOT.h"
#include <iostream>

class PixelROCGainCalibElement : public TObject
{
 private :

  float  fResponse;//
  float  fTotVal;
  unsigned int fntimes;//
  unsigned int fvcalval;//

 public :

  PixelROCGainCalibElement():fResponse(0.),fTotVal(0.),fntimes(0),fvcalval(0){;}

    virtual ~PixelROCGainCalibElement(){;}

 //- Accessible methods
  //- Object Status


  void Clear(Option_t* = "") {;}
  void Reset(Option_t* = "") {;}

  void Print(Option_t* = "") const {;}

 //-- Setter/Getter
  
  void addValue(const unsigned int & in ) { fResponse+=in ; ++fntimes; fTotVal=fResponse/(float)fntimes;}
  float getValue() const { return fTotVal;}
  void setVCalValue(const unsigned int & in) { fvcalval = in; }
  unsigned int getVCalValue() const {return fvcalval;}
  unsigned int  getNtimes() const { return fntimes;}


};

#endif

