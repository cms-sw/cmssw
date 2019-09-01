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

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include <tuple>

void SiPixelGainCalibrationForHLTService::calibrate(
    uint32_t detID, DigiIterator b, DigiIterator e, float conversionFactor, float offset, int* electron) {
  SiPixelGainCalibrationForHLT::Range range;
  int cols;
  std::tie(range, cols) = ped->getRangeAndNCols(detID);
  float pedestal = 0, gain = 0;
  int i = 0;
  bool isDeadColumn = false, isNoisyColumn = false;
  int oldCol = -1, oldAveragedBlock = -1;
  for (DigiIterator di = b; di != e; ++di) {
    int row = di->row();
    int col = di->column();
    int averagedBlock = row / numberOfRowsAveragedOver_;
    if ((col != oldCol) | (averagedBlock != oldAveragedBlock)) {
      oldCol = col;
      oldAveragedBlock = averagedBlock;
      std::tie(pedestal, gain) = ped->getPedAndGain(col, row, range, cols, isDeadColumn, isNoisyColumn);
    }
    if (isDeadColumn | isNoisyColumn)
      electron[i++] = 0;
    else {
      float vcal = di->adc() * gain - pedestal * gain;
      //    float vcal = (di->adc()  - DBpedestal) * DBgain;
      electron[i++] = int(vcal * conversionFactor + offset);
    }
  }
  assert(i == (e - b));
}

float SiPixelGainCalibrationForHLTService::getPedestal(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  float pedestalValue = this->getPedestalByColumn(detID, col, row, isDead, isNoisy);
  if (isDead || isNoisy) {
    this->throwExepctionForBadRead("HLT getPedestal()", detID, col, row, pedestalValue);
    return 0.0;
  }
  return pedestalValue;
}

float SiPixelGainCalibrationForHLTService::getGain(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  float gainValue = this->getGainByColumn(detID, col, row, isDead, isNoisy);
  if (isDead || isNoisy) {
    this->throwExepctionForBadRead("HLT getGain()", detID, col, row, gainValue);
    return 0.0;
  }
  return gainValue;
}

bool SiPixelGainCalibrationForHLTService::isDead(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  try {
    this->getPedestalByColumn(detID, col, row, isDead, isNoisy);  //pedestal stores dead column value as well
  } catch (cms::Exception& e) {
    // Do not stop processing if you check if a nonexistant pixel is dead
    edm::LogInfo("SiPixelGainCalibrationForHLTService")
        << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
    isDead = false;
  }
  return isDead;
}

bool SiPixelGainCalibrationForHLTService::isNoisy(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  try {
    this->getPedestalByColumn(detID, col, row, isDead, isNoisy);  //pedestal stores noisy column value as well
  } catch (cms::Exception& e) {
    // Do not stop processing if you check if a nonexistant pixel is noisy
    edm::LogInfo("SiPixelGainCalibrationForHLTService")
        << "Attempting to check if nonexistant pixel is noisy.  Exception message: " << e.what();
    isNoisy = false;
  }
  return isNoisy;
}

bool SiPixelGainCalibrationForHLTService::isDeadColumn(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  try {
    this->getGainByColumn(detID, col, row, isDead, isNoisy);
  } catch (cms::Exception& e) {
    // Do not stop processing if you check if a nonexistant pixel is dead
    edm::LogInfo("SiPixelGainCalibrationForHLTService")
        << "Attempting to check if nonexistant pixel is dead.  Exception message: " << e.what();
    isDead = false;
  }
  return isDead;
}

bool SiPixelGainCalibrationForHLTService::isNoisyColumn(const uint32_t& detID, const int& col, const int& row) {
  bool isDead = false;
  bool isNoisy = false;
  try {
    this->getGainByColumn(detID, col, row, isDead, isNoisy);
  } catch (cms::Exception& e) {
    // Do not stop processing if you check if a nonexistant pixel is noisy
    edm::LogInfo("SiPixelGainCalibrationForHLTService")
        << "Attempting to check if nonexistant pixel is noisy.  Exception message: " << e.what();
    isNoisy = false;
  }
  return isNoisy;
}
