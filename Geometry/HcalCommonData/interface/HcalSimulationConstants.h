#ifndef Geometry_HcalCommonData_HcalSimulationConstants_h
#define Geometry_HcalCommonData_HcalSimulationConstants_h

/** \class HcalSimulationConstants
 *
 * this class reads the constant section of
 * the xml-files related to HF for HCAL simulation
 *  
 * \author Sunanda Banerjee, FNAL <sunanda.banerjee@cern.ch>
 *
 */

#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"

class HcalSimulationConstants {
public:
  HcalSimulationConstants(const HcalSimulationParameters* hps);
  ~HcalSimulationConstants();

  const HcalSimulationParameters* hcalsimpar() const { return hspar_; }

private:
  const HcalSimulationParameters* hspar_;
};

#endif
