/*
 * =====================================================================================
 *
 *       Filename:  SiPixelGainCalibrationOfflineServic.cc
 *
 *    Description:  
 *
 *        Version:  1.0 (some functionality moved from ../interface/SiPixelGainCalibrationOfflineService.h)
 *        Created:  04/16/2008 10:35:35 AM
 *
 *         Author:  Evan Friis (evan.klose.friis@cern.ch)
 *                  University of California, Davis
 *
 * =====================================================================================
 */

#include "CondTools/SiPixel/interface/SiPixelGainCalibrationOfflineService.h"

float SiPixelGainCalibrationOfflineService::getPedestal( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   float pedestalValue = this->getPedestalByPixel(detID, col, row, isDead);
   if (isDead)
      this->throwExepctionForBadRead("Offline", detID, col, row);
   return pedestalValue;
}

float SiPixelGainCalibrationOfflineService::getGain( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   float gainValue = this->getGainByColumn(detID, col, row, isDead);
   if (isDead)
      this->throwExepctionForBadRead("Offline", detID, col, row);
   return gainValue;
}

bool SiPixelGainCalibrationOfflineService::isDead( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   this->getPedestalByPixel(detID, col, row, isDead); 
   return isDead;
}
   
bool SiPixelGainCalibrationOfflineService::isDeadColumn( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   this->getGainByColumn(detID, col, row, isDead); // the gain column average can flag a whole column as bad
   return isDead;
}

