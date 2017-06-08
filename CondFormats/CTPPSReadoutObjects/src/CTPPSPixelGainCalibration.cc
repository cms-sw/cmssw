#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <cstring>

//
// Constructors
//
CTPPSPixelGainCalibration::CTPPSPixelGainCalibration() :
  minPed_(0.),
  maxPed_(255.),
  minGain_(0.),
  maxGain_(255.)
{
  DetRegistry myreg;
  myreg.detid=0;
  myreg.ibegin=0;
  myreg.iend=0;
  myreg.ncols=0; 
  indexes = myreg;
  edm::LogInfo("CTPPSPixelGainCalibration")<<"Default instance of CTPPSPixelGainCalibration";
}

//
CTPPSPixelGainCalibration::CTPPSPixelGainCalibration(float minPed, float maxPed, float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain) 
{
  DetRegistry myreg;
  myreg.detid=0;
  myreg.ibegin=0;
  myreg.iend=0;
  myreg.ncols=0; 
  indexes = myreg;
  edm::LogInfo("CTPPSPixelGainCalibration")<<"Instance of CTPPSPixelGainCalibration setting mininums and maximums";
}

CTPPSPixelGainCalibration::CTPPSPixelGainCalibration(const uint32_t& detid, const std::vector<float> & peds, const std::vector<float>& gains, float minPed, float maxPed,float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain)
{
  edm::LogInfo("CTPPSPixelGainCalibration")<<"Instance of CTPPSPixelGainCalibration setting peds and gains";
  setGainsPeds(detid,peds,gains);
}

void CTPPSPixelGainCalibration::setGainsPeds(const uint32_t& detid, const std::vector<float> & peds, const std::vector<float>& gains){
  int sensorSize=peds.size();
  int gainsSize=gains.size();
  if(gainsSize!=sensorSize)
    throw cms::Exception("CTPPSPixelGainCalibration payload configuration error") 
      <<  "[CTPPSPixelGainCalibration::CTPPSPixelGainCalibration]  peds and gains vector sizes don't match for detid "
      << detid << " size peds = " << sensorSize << " size gains = " << gainsSize; 
  DetRegistry myreg;
  myreg.detid=detid;
  myreg.ibegin=0;
  myreg.iend=sensorSize;
  myreg.ncols=sensorSize/160; // each ROC is made of 80 rows and 52 columns, each sensor is made of 160 rows and either 104 or 156 columns (2x2 or 2x3 ROCs)
  indexes = myreg;
  for(int i = 0 ; i<sensorSize ; ++i)  
    putData(i,peds[i],gains[i]);
}

void CTPPSPixelGainCalibration::putData(uint32_t ipix, float ped, float gain){
  if (v_pedestals.size()>ipix) //the vector is too big, this pixel has already been set
    {
      if (ped>=0 && gain>=0)
	throw cms::Exception("CTPPSPixelGainCalibration fill error")
	  << "[CTPPSPixelGainCalibration::putData] attemptint to fill the vectors that are already filled detid = " << indexes.detid << " ipix " << ipix;
      else   // in case it is for setting the noisy or dead flag of already filled pixel
	{
	  edm::LogInfo("CTPPSPixelGainCalibration")<<"resetting pixel calibration for noisy or dead flag";
	  resetPixData(ipix,ped,gain);
	}
    }
  else if (v_pedestals.size()<ipix)
    throw cms::Exception("CTPPSPixelGainCalibration fill error")
      << "[CTPPSPixelGainCalibration::putData] the order got scrambled detid = "<< indexes.detid << " ipix " << ipix;
  else{ //the size has to match exactly the index, the index - 0 pixel starts the vector, the one with index 1 should be pushed back in a vector of size== 1 (to become size==2) so on and o forth
    v_pedestals.push_back(ped);
    v_gains.push_back(gain);
  }
}
  
void CTPPSPixelGainCalibration::resetPixData(uint32_t ipix, float ped, float gain){
  if (v_pedestals.size()<=ipix){
    edm::LogError("CTPPSPixelGainCalibration")<<"ERROR here this element has not been set yet"; 
    return;
  }
  v_pedestals[ipix]=ped; //modify value to already exisiting element
  v_gains[ipix]=gain;
}


float CTPPSPixelGainCalibration::getPed(const int& col, const int& row) const {

  int nCols=indexes.ncols;
  int nRows=v_pedestals.size()/nCols;
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[CTPPSPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  int ipix = col + row * nCols;
  return   getPed(ipix);
}


float CTPPSPixelGainCalibration::getPed(const uint32_t ipix) const {

  return v_pedestals[ipix];  

}




float CTPPSPixelGainCalibration::getGain(const int& col, const int& row) const {

  int nCols=indexes.ncols;
  int nRows=v_pedestals.size()/nCols;
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[CTPPSPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  int ipix = col + row * nCols;
  return getGain(ipix);
}

float CTPPSPixelGainCalibration::getGain(const uint32_t ipix) const {

  return v_gains[ipix];

}


