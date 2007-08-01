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
  unsigned int fvcalval;//

 public :

  PixelROCGainCalibElement():fResponse(0.),fvcalval(0){;}

    virtual ~PixelROCGainCalibElement(){;}

 //- Accessible methods
  //- Object Status


  void Clear(Option_t* = "") {;}
  void Reset(Option_t* = "") {;}

  void Print(Option_t* = "") const {;}

 //-- Setter/Getter
  
  void addValue( unsigned int  in ) {fResponse+=in ;}
  //  { std::cout << "addvalue: " << fResponse << " " << in << ", new "; fResponse+=in ; ++fntimes; std::cout << fResponse << " " << fntimes << std::endl;}
  float getValue() { return fResponse;}
  void setVCalValue(const unsigned int & in) { fvcalval = in; }
  unsigned int getVCalValue() const {return fvcalval;}


};

#endif

