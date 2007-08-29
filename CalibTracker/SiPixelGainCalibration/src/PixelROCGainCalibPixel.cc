#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <math>

double PixelROCGainCalibPixel::geterror(uint32_t icalpoint, uint32_t ntimes){
  double res=0;
  
  if(nentries[icalpoint]>0){
    double meansq = adcvalues[icalpoint]*adcvalues[icalpoint];
    res = fabs( meansq - sumsquares[icalpoint] );
    res /= (double) nentries[icalpoint];
  }
  return sqrt(res);
}
double PixelROCGainCalibPixel::getpoint(uint32_t icalpoint, uint32_t ntimes){
  //  std::cout << "getting point " << icalpoint << ", value " << adcvalues[icalpoint] << std::endl;
  double res=0.;
  res+=adcvalues[icalpoint];
  double denom = nentries[icalpoint];
  if(nentries[icalpoint]!=ntimes && ntimes>0)
    edm::LogError("PixelROCGainCalibPixel") << " error: expected "<< ntimes << " points, but only received " << nentries[icalpoint] << " point" << std::endl;
  if(denom>1){
    res/=denom;
  }
  return res;
}
PixelROCGainCalibPixel::PixelROCGainCalibPixel(uint32_t npoints):adcvalues(npoints,0),nentries(npoints,0),sumsquares(npoints,0){
}

PixelROCGainCalibPixel::~PixelROCGainCalibPixel(){
  adcvalues.clear();
}
void PixelROCGainCalibPixel::clearAllPoints(){
  for(uint32_t i=0; i<adcvalues.size(); i++){
    adcvalues[i]=0;
    nentries[i]=0;
    sumsquares[i]=0;
  }
}
//******************
bool PixelROCGainCalibPixel::isfilled(){
  for(uint32_t i=0; i< adcvalues.size(); i++){
    if(nentries[i]>0)
      return true;
  }
  return false;
}
//******************
void PixelROCGainCalibPixel::init(uint32_t nvcal)
{
   adcvalues.resize(nvcal,0);  
   nentries.resize(nvcal,0);
   sumsquares.resize(nvcal,0);
}

//******************
void PixelROCGainCalibPixel::addPoint(uint32_t icalpoint, uint32_t adcval){
  if(icalpoint>adcvalues.size()){
    //    edm::LogInfo("ERROR")
    std::cout <<"ERROR: point "<< icalpoint << " > size  " << adcvalues.size() << " " << nentries.size() << std::endl;
    return;
  }
  //  std::cout << icalpoint << " " << adcvalues[icalpoint] << " + " << adcval << " = ";
  adcvalues[icalpoint] += adcval;
  sumsquares[icalpoint] += adcval*adcval;
  nentries[icalpoint]++;
  
}
//******************

TGraph* PixelROCGainCalibPixel::createGraph(TString thename, TString thetitle, const int ntimes,std::vector<uint32_t> vcalvalues){
  TGraph *result = new TGraph(vcalvalues.size());
  for(uint32_t i=0;i<vcalvalues.size() && ntimes>0;++i){
    edm::LogInfo("DEBUG") <<adcvalues[i] << " " << ntimes << " " << i << std::endl;
    float val = adcvalues[i];
    val/=(float)nentries[i];
    
    result->SetPoint(i,vcalvalues[i],val);
  }
  result->GetHistogram()->GetXaxis()->SetTitle("VCAL values");
  result->GetHistogram()->GetYaxis()->SetTitle("ADC counts");
  result->SetName(thename);
  result->GetHistogram()->SetTitle(thetitle);
  return result;
}

