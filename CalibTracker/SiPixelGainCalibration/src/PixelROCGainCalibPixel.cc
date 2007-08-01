#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"

double PixelROCGainCalibPixel::getpoint(uint32_t icalpoint, uint32_t ntimes){
  //  std::cout << "getting point " << icalpoint << ", value " << adcvalues[icalpoint] << std::endl;
  double res=0.;
  res+=adcvalues[icalpoint];
  if(ntimes>1){
    double denom=ntimes;
    res/=denom;
  }
  return res;
}
PixelROCGainCalibPixel::PixelROCGainCalibPixel(uint32_t npoints):adcvalues(npoints,0){
}

PixelROCGainCalibPixel::~PixelROCGainCalibPixel(){
  adcvalues.clear();
}

//******************
bool PixelROCGainCalibPixel::isfilled(){
  for(uint32_t i=0; i< adcvalues.size(); i++){
    if(adcvalues[i]>0)
      return true;
  }
  return false;
}
//******************
void PixelROCGainCalibPixel::init(uint32_t nvcal)
{
   adcvalues.resize(nvcal,0);  
}

//******************
void PixelROCGainCalibPixel::addPoint(uint32_t icalpoint, uint32_t adcval){
  if(icalpoint>adcvalues.size()){
    //    edm::LogInfo("ERROR")
    std::cout <<"ERROR: point "<< icalpoint << " > size  " << adcvalues.size() << std::endl;
    return;
  }
  //  std::cout << icalpoint << " " << adcvalues[icalpoint] << " + " << adcval << " = ";
  uint32_t newval = adcvalues[icalpoint];
  newval+=adcval;
  adcvalues[icalpoint] = newval;

//   std::cout << adcvalues[icalpoint] << ", size ";
//   std::cout << adcvalues.size();
//   std::cout <<" other points: ";
//   for(int ip=0; ip<adcvalues.size();ip++)
//     if(ip!=icalpoint)
//       std::cout << adcvalues[ip] << " ";
//   std::cout << std::endl;
  
  
}
//******************

TGraph* PixelROCGainCalibPixel::createGraph(TString thename, TString thetitle, const int ntimes,std::vector<uint32_t> vcalvalues){
  TGraph *result = new TGraph(vcalvalues.size());
  for(uint32_t i=0;i<vcalvalues.size() && ntimes>0;++i){
    edm::LogInfo("DEBUG") <<adcvalues[i] << " " << ntimes << " " << i << std::endl;
    float val = adcvalues[i];
    val/=(float)ntimes;
    
    result->SetPoint(i,vcalvalues[i],val);
  }
  result->GetHistogram()->GetXaxis()->SetTitle("VCAL values");
  result->GetHistogram()->GetYaxis()->SetTitle("ADC counts");
  result->SetName(thename);
  result->GetHistogram()->SetTitle(thetitle);
  return result;
}

