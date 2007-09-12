//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : PixelROCGainCalibPixel

//---------------------------------------------------
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <math>

double PixelROCGainCalibPixel::geterror(uint32_t icalpoint, uint32_t ntimes){
  double res=0;
  
  if(nentries[icalpoint]>0){
    double meansq = (adcvalues[icalpoint]*adcvalues[icalpoint])*1.0/(nentries[icalpoint]*nentries[icalpoint]);
    res = fabs( meansq - (sumsquares[icalpoint]*1.0/nentries[icalpoint] ));
    //res /= (double) nentries[icalpoint];
  }
  return sqrt(res);
}
double PixelROCGainCalibPixel::getefficiency(uint32_t icalpoint, uint32_t ntimes){
  double res = nentries[icalpoint];
  if(ntimes>0)
    res /= (double) ntimes;
  else
    res = -1;
  return res;
}

double PixelROCGainCalibPixel::geterroreff(uint32_t icalpoint, uint32_t ntimes){
  double m = nentries[icalpoint];
  double n = ntimes;
  double response = getefficiency(icalpoint,ntimes);
  
  double d = n *n *(1+2.*m)-4.*n*(1.+n)*m*m;
  //only fill the largest of the two (asymmetric) binomial errors
  double temperror = ( n*(1.0+2.0*m) + sqrt(d) )/(2.0*n*(1.0+n));
  double result = fabs(temperror-response);
  temperror = ( n*(1.0+2.0*m) - sqrt(d) )/(2.0*n*(1.0+n));
  if(fabs(temperror-response)>result)
    result = fabs(temperror-response);
  return result;
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

void PixelROCGainCalibPixel::doAnalyticalFit(std::vector<uint32_t> vcalvalues, uint32_t vcallow, uint32_t vcalhigh){
  // does an analytical fit, original code from Tony Kelly
  assert(vcalvalues.size()==nentries.size());
  double xpoints_mean_sum=0;
  double xpoints_sqmean_sum=0;
  double ypoints_mean_sum=0;
  double ypoints_sqmean_sum=0;
  double xpoints_mean=0;
  double xpoints_sqmean=0;
  double xpoints_meansq=0;
  double ypoints_mean=0;
  double ypoints_sqmean=0;
  double ypoints_meansq=0;
  //The following variables are for the regression line, defaulted to a horizontal line
  double slope_numerator=0;
  double slope_denominator=1;
  double slope = 0;
  double intercept=0;
  double regression_ydenominator=1;
  // regression is necesary for error calculation but not included yet.
  //double regression=0;
  //  double regression_square=0;

  double npoints_used=0;
  for(uint32_t ipoint = 0; ipoint < nentries.size(); ++ipoint){
    if(vcalvalues[ipoint]< vcallow)
      continue;
    if(vcalvalues[ipoint]>vcalhigh)
      continue;
    xpoints_mean_sum += vcalvalues[ipoint];
    ypoints_mean_sum += getpoint(ipoint,getentries(ipoint));   
    xpoints_sqmean_sum += (vcalvalues[ipoint]*vcalvalues[ipoint]);
    ypoints_sqmean_sum += getpoint(ipoint,getentries(ipoint))*getpoint(ipoint,getentries(ipoint));
    npoints_used++;
  }
  if(npoints_used==0){
    pedanderror.first=pedanderror.second = gainanderror.first = gainanderror.second =-1;
    return;
  }
    
  xpoints_mean = xpoints_mean_sum/npoints_used;
  xpoints_sqmean=xpoints_sqmean_sum/npoints_used;
  xpoints_meansq=xpoints_mean*xpoints_mean;
  ypoints_mean = ypoints_mean_sum/npoints_used;
  ypoints_sqmean=ypoints_sqmean_sum/npoints_used;
  ypoints_meansq=ypoints_mean*ypoints_mean;

  for(uint32_t ipoint = 0; ipoint < nentries.size(); ++ipoint){
    if(vcalvalues[ipoint]< vcallow)
      continue;
    if(vcalvalues[ipoint]>vcalhigh)
      continue;
    slope_numerator += (vcalvalues[ipoint]-xpoints_mean)*(getpoint(ipoint,getentries(ipoint))-ypoints_mean);
    slope_denominator += (vcalvalues[ipoint]-xpoints_mean)*(vcalvalues[ipoint]-xpoints_mean);
    regression_ydenominator += (getpoint(ipoint,getentries(ipoint))-ypoints_mean)*(getpoint(ipoint,getentries(ipoint))-ypoints_mean);
  }
  slope = slope_numerator/slope_denominator;
  
  gainanderror.first = slope;
  intercept = ypoints_mean-(slope*xpoints_mean);
  pedanderror.first=intercept;

  // regression is necesary for error calculation but not included yet.
  //  regression= (slope_numerator)/(TMath::Sqrt(slope_denominator*regression_ydenominator));
  //  regression_square=regression*regression;

  
}
TGraph* PixelROCGainCalibPixel::createGraph(TString thename, TString thetitle, const int ntimes,std::vector<uint32_t> vcalvalues){
  TGraph *result = new TGraph(vcalvalues.size());
  for(uint32_t i=0;i<vcalvalues.size() && ntimes>0;++i){
    edm::LogInfo("DEBUG") <<adcvalues[i] << " " << ntimes << " " << i << std::endl;
    double val = adcvalues[i];
    val/=(double)nentries[i];
    
    result->SetPoint(i,vcalvalues[i],val);
  }
  result->GetHistogram()->GetXaxis()->SetTitle("VCAL values");
  result->GetHistogram()->GetYaxis()->SetTitle("ADC counts");
  result->SetName(thename);
  result->GetHistogram()->SetTitle(thetitle);
  return result;
}

