/*
 * =====================================================================================
 *
 *       Filename:  SiPixelGainCalibrationServic.cc
 *
 *    Description:  
 *
 *        Version:  1.0 (some functionality moved from ../interface/SiPixelGainCalibrationService.h)
 *        Created:  04/16/2008 10:35:35 AM
 *
 *         Author:  Evan Friis (evan.klose.friis@cern.ch)
 *                  University of California, Davis
 *
 * =====================================================================================
 */

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"

float SiPixelGainCalibrationService::getPedestal( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   float pedestalValue = this->getPedestalByPixel(detID, col, row, isDead, isNoisy);
   if (isDead || isNoisy)
   {
      this->throwExepctionForBadRead("FullCalibration getPedestal()",detID, col, row, pedestalValue);
      return 0.0;
   }
   return pedestalValue;
}

float SiPixelGainCalibrationService::getGain( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   float gainValue = this->getGainByColumn(detID, col, row, isDead, isNoisy);
   if (isDead || isNoisy)
   {
      this->throwExepctionForBadRead("FullCalibration getGain()",detID, col, row, gainValue);
      return 0.0;
   }
   return gainValue;
}

bool SiPixelGainCalibrationService::isDead( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getPedestalByPixel(detID, col, row, isDead, isNoisy); 
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is dead
      edm::LogInfo("SiPixelGainCalibrationService") << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
      isDead = false; 
   }
   return isDead;
}

bool SiPixelGainCalibrationService::isNoisy( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getPedestalByPixel(detID, col, row, isDead, isNoisy); 
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is noisy
      edm::LogInfo("SiPixelGainCalibrationService") << "Attempting to check if nonexistant pixel is noisy.  Exception message: " << e.what();
      isNoisy = false; 
   }
   return isNoisy;
}
   
bool SiPixelGainCalibrationService::isDeadColumn( const uint32_t& detID,const int& col, const int& row)
{
   edm::LogError("SiPixelGainCalibrationService") << "You attempted to check if an entire column was dead with a payload that stores information at pixel granularity.  Please check by pixel. THANKS!";
   return false;
}
   
bool SiPixelGainCalibrationService::isNoisyColumn( const uint32_t& detID,const int& col, const int& row)
{
   edm::LogError("SiPixelGainCalibrationService") << "You attempted to check if an entire column was noisy with a payload that stores information at pixel granularity.  Please check by pixel. THANKS!";
   return false;
}
