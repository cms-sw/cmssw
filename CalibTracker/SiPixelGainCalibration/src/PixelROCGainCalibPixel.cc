#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"
#include <iostream>
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"

PixelROCGainCalibPixel::PixelROCGainCalibPixel(){

  calibPoints = new TObjArray(10);
 }

//******************
void PixelROCGainCalibPixel::init(unsigned int nvcal)
{
  for(unsigned int i=0; i<nvcal; i++){
    PixelROCGainCalibElement *ele = new PixelROCGainCalibElement();
    calibPoints->Add(ele);
  }
}

//******************
void PixelROCGainCalibPixel::addPoint(unsigned int icalpoint, unsigned int adcval, unsigned int vcalval){
  if(icalpoint<calibPoints->GetEntries() && icalpoint>=0){
    PixelROCGainCalibElement* ele = (PixelROCGainCalibElement*)calibPoints->At(icalpoint);
    //    std::cout << "value now: " << ele->getValue() << " at icalpoint " << icalpoint << " " << ele->getVCalValue() << std::endl;
    if(ele->getVCalValue()==0)
      ele->setVCalValue(vcalval);
    ele->addValue(adcval);
    //    std::cout << "value now: " << ele->getValue() << " at icalpoint " << icalpoint << " " << ele->getVCalValue() << std::endl;
  }
}
//******************

TH1F* PixelROCGainCalibPixel::createHistogram(TString thename, TString thetitle, int nbins, float lowval, float highval){
  TH1F* result = new TH1F(thename,thetitle,nbins,lowval,highval);
  result->GetXaxis()->SetTitle("VCAL values");
  result->GetYaxis()->SetTitle("ADC counts");
  for(int i=0; i<=calibPoints->GetEntries();++i){
    PixelROCGainCalibElement * ele = (PixelROCGainCalibElement*) calibPoints->At(i);
    if(!ele)
      continue;
    
    if( ele->getVCalValue()>0 && ele->getVCalValue()<=256 && ele->getValue()>0 && ele->getValue()<=256){
      //      std::cout << "filling histogram with : " << ele->getVCalValue() << " " << ele->getValue() << std::endl;
      result->Fill(ele->getVCalValue(),ele->getValue());
    }
  }
  return result;
}

