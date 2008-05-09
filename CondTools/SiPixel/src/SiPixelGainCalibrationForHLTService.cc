/*
 * =====================================================================================
 *
 *       Filename:  SiPixelGainCalibrationForHLT.cc
 *
 *    Description:  
 *
 *        Version:  1.0 (some functionality moved from ../interface/SiPixelGainCalibrationForHLT.h)
 *        Created:  04/16/2008 10:35:35 AM
 *
 *         Author:  Evan Friis (evan.klose.friis@cern.ch)
 *                  University of California, Davis
 *
 * =====================================================================================
 */

#include "CondTools/SiPixel/interface/SiPixelGainCalibrationForHLTService.h"

float SiPixelGainCalibrationForHLTService::getPedestal( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   float pedestalValue = this->getPedestalByColumn(detID, col, row, isDead);
   if (isDead)
   {
      this->throwExepctionForBadRead("HLT getPedestal()", detID, col, row, pedestalValue);
      return 0.0;
   }
   return pedestalValue;
}

float SiPixelGainCalibrationForHLTService::getGain( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   float gainValue = this->getGainByColumn(detID, col, row, isDead);
   if (isDead)
   {
      this->throwExepctionForBadRead("HLT getGain()", detID, col, row, gainValue);
      return 0.0;
   }
   return gainValue;
}

bool SiPixelGainCalibrationForHLTService::isDead( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   try  
   {
      this->getPedestalByColumn(detID, col, row, isDead); //pedestal stores dead column value as well
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is dead
      edm::LogInfo("SiPixelGainCalibrationForHLTService") << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
      isDead = false; 
   }
   return isDead;
}
   
bool SiPixelGainCalibrationForHLTService::isDeadColumn( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   try  
   {
      this->getGainByColumn(detID, col, row, isDead);
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is dead
      edm::LogInfo("SiPixelGainCalibrationForHLTService") << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
      isDead = false; 
   }
   return isDead;
}

