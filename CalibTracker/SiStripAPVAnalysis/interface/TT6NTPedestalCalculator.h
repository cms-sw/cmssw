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
class TT6NTPedestalCalculator: public TkPedestalCalculator
{
  public:
    TT6NTPedestalCalculator();
    virtual ~TT6NTPedestalCalculator() {}

    /*
     * @brief
     *   Celar all Pedestals
     */
    inline virtual void resetPedestals() { pedestals_.empty(); }

    /*
     * @brief
     *   Set Pedestals
     */
    inline virtual void setPedestals( ApvAnalysis::PedestalType &rInput) 
      { pedestals_ = rInput; }

    /*
     * @brief
     *   Update Pedestals with set of Raw Signals: plug
     */
    inline virtual void updatePedestal( ApvAnalysis::RawSignalType &rInput) {}

    /*
     * @brief
     *   Retrieve Pedestals
     */
    inline virtual ApvAnalysis::PedestalType pedestal() const { return pedestals_; }

    /*
     * @brief
     *   Retrieve Raw Noise
     */
    inline virtual ApvAnalysis::PedestalType rawNoise() const { return rawNoise_; }

    inline virtual void setNoise( ApvAnalysis::PedestalType &rInput)
      { rawNoise_ = rInput; }

    /*
     * @brief
     *   Request status flag update: plug
     */
    inline virtual void updateStatus() {}

  private:
    ApvAnalysis::PedestalType pedestals_;
    ApvAnalysis::PedestalType rawNoise_;
};

#endif // APVANALYSIS_TT6NTPEDESTALCALCULATOR_H
