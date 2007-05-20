//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelSLinkDataHit

//---------------------------------------------------

#ifndef PixelSLinkDataHit_H
#define PixelSLinkDataHit_H


#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class PixelSLinkDataHit 
{
 private :

 
  unsigned long long  fTheData; // the actual data 
  unsigned int  fTheHitData; // the data filtered for a hit
  bool fHitSet; // check that the we're looking at either first or second hit
  bool fValidChecked; // for dummy users: will eventually be set so you can only get data if this bool is set.
 public :

  PixelSLinkDataHit() {;}

  virtual ~PixelSLinkDataHit(){;}

 //- Accessible methods
  void SetTheData(unsigned long long data){fTheData = data; fHitSet=false;}
  void SetHit(unsigned int ihit);
  //- Object Status


 //-- Setter/Getter

  bool IsValidData();
  bool IsValidHit();

  unsigned int  GetFEDChannel() const { return (fTheHitData>>26)&0x3f;}
  unsigned int  GetROC_ID() const { return (fTheHitData>>21)&0x1f;} 
  unsigned int  GetDcol_ID() const { return (fTheHitData>>16)&0x1f;}
  unsigned int  GetPix_ID() const { return (fTheHitData>>8)&0xff;}
  unsigned int  GetADC_count() const { return fTheHitData&0xff;}
  unsigned int  GetPixelRow() const { return 80-GetPix_ID()/2;}
  unsigned int  GetPixelColumn() const {  return GetDcol_ID()*2 +GetPix_ID()%2; }
  
};

inline bool PixelSLinkDataHit::IsValidData(void){
  
  if((fTheData >>60) == 0x5) return false; 
  if((fTheData >> 60) == 0xa) return false;
  if(fTheHitData==0xffffffff && fHitSet)
    return false;
  if(fTheHitData==1 && fHitSet)
    return false;
  return true;
}

inline void PixelSLinkDataHit::SetHit(unsigned int ihit){ 
  assert(ihit<2); 
  assert(ihit>=0);
  fTheHitData = (fTheData>>(ihit*32))&(0xffffffff);
  fHitSet=true;
  return;
 }

inline bool PixelSLinkDataHit::IsValidHit(void){
  if( GetPixelRow()>=80 || GetPixelColumn() >=52 || GetROC_ID() > 24 || GetFEDChannel() > 36 ) {  
    edm::LogVerbatim log("");
    log << " INVALID HIT: " << std::endl;
    log << " hitdata="<<std::hex<<fTheHitData<<std::dec
	<< " fed_channel="<<GetFEDChannel()
	<< " roc_id="<<GetROC_ID()
	<< " pix_id="<<GetPix_ID()
	<< " dcol_id="<<GetDcol_ID();
    log << " row="<<GetPixelRow()
	<< " col="<<GetPixelColumn()
	<< " adc="<<GetADC_count()
	<< std::endl;
    //  throw cms::Exception("AssertFailure")<<"Invalid pixel hit ";
    return false;
  }
  return true;
}
#endif

