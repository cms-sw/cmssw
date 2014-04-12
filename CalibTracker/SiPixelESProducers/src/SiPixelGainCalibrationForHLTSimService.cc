/*
 * =====================================================================================
 *
 *       Filename:  SiPixelGainCalibrationForHLTSim.cc
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

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTSimService.h"

float SiPixelGainCalibrationForHLTSimService::getPedestal( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   float pedestalValue = this->getPedestalByColumn(detID, col, row, isDead, isNoisy);
   if (isDead || isNoisy)
   {
      this->throwExepctionForBadRead("HLT getPedestal()", detID, col, row, pedestalValue);
      return 0.0;
   }
   return pedestalValue;
}

float SiPixelGainCalibrationForHLTSimService::getGain( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   float gainValue = this->getGainByColumn(detID, col, row, isDead, isNoisy);
   if (isDead || isNoisy)
   {
      this->throwExepctionForBadRead("HLT getGain()", detID, col, row, gainValue);
      return 0.0;
   }
   return gainValue;
}

bool SiPixelGainCalibrationForHLTSimService::isDead( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getPedestalByColumn(detID, col, row, isDead, isNoisy); //pedestal stores dead column value as well
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is dead
      edm::LogInfo("SiPixelGainCalibrationForHLTSimService") << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
      isDead = false; 
   }
   return isDead;
}

bool SiPixelGainCalibrationForHLTSimService::isNoisy( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getPedestalByColumn(detID, col, row, isDead, isNoisy); //pedestal stores noisy column value as well
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is noisy
      edm::LogInfo("SiPixelGainCalibrationForHLTSimService") << "Attempting to check if nonexistant pixel is noisy.  Exception message: " << e.what();
      isNoisy = false; 
   }
   return isNoisy;
}
   
bool SiPixelGainCalibrationForHLTSimService::isDeadColumn( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getGainByColumn(detID, col, row, isDead, isNoisy);
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is dead
      edm::LogInfo("SiPixelGainCalibrationForHLTSimService") << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
      isDead = false; 
   }
   return isDead;
}
  
bool SiPixelGainCalibrationForHLTSimService::isNoisyColumn( const uint32_t& detID,const int& col, const int& row)
{
   bool isDead = false;
   bool isNoisy = false;
   try  
   {
      this->getGainByColumn(detID, col, row, isDead, isNoisy);
   }
   catch (cms::Exception& e) 
   {
      // Do not stop processing if you check if a nonexistant pixel is noisy
      edm::LogInfo("SiPixelGainCalibrationForHLTSimService") << "Attempting to check if nonexistant pixel is noisy.  Exception message: " << e.what();
      isNoisy = false; 
   }
   return isNoisy;
}

