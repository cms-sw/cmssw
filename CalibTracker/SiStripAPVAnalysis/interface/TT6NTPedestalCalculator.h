// Author : Samvel Khalatian (samvel at cern dot ch)
// Created: 05/21/08

#ifndef APVANALYSIS_TT6NTPEDESTALCALCULATOR_H
#define APVANALYSIS_TT6NTPEDESTALCALCULATOR_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"

/*
 * @brief
 *   This is a replacement of TT6 Pedestal Calculator for NoiseTask source in
 *   DQM/SiStripCommissioningSources. It's main tasks:
 *     1. Retrieve Pedestals from DB
 *     2. Return these Pedestals on demand
 *   Note: no additional calculations performed
 */
class TT6NTPedestalCalculator : public TkPedestalCalculator {
public:
  TT6NTPedestalCalculator();
  ~TT6NTPedestalCalculator() override {}

  /*
     * @brief
     *   Celar all Pedestals
     */
  inline void resetPedestals() override { pedestals_.clear(); }

  /*
     * @brief
     *   Set Pedestals
     */
  inline void setPedestals(ApvAnalysis::PedestalType &rInput) override { pedestals_ = rInput; }

  /*
     * @brief
     *   Update Pedestals with set of Raw Signals: plug
     */
  inline void updatePedestal(ApvAnalysis::RawSignalType &rInput) override {}

  /*
     * @brief
     *   Retrieve Pedestals
     */
  inline ApvAnalysis::PedestalType pedestal() const override { return pedestals_; }

  /*
     * @brief
     *   Retrieve Raw Noise
     */
  inline ApvAnalysis::PedestalType rawNoise() const override { return rawNoise_; }

  inline void setNoise(ApvAnalysis::PedestalType &rInput) override { rawNoise_ = rInput; }

  /*
     * @brief
     *   Request status flag update: plug
     */
  inline void updateStatus() override {}

private:
  ApvAnalysis::PedestalType pedestals_;
  ApvAnalysis::PedestalType rawNoise_;
};

#endif  // APVANALYSIS_TT6NTPEDESTALCALCULATOR_H
