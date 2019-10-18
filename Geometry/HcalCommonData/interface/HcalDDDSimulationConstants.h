#ifndef Geometry_HcalCommonData_HcalDDDSimulationConstants_h
#define Geometry_HcalCommonData_HcalDDDSimulationConstants_h

/** \class HcalDDDSimulationConstants
 *
 * this class reads the constant section of
 * the xml-files related to HF for HCAL simulation
 *  
 * \author Sunanda Banerjee, FNAL <sunanda.banerjee@cern.ch>
 *
 */

#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"

class HcalDDDSimulationConstants {
public:
  HcalDDDSimulationConstants(const HcalSimulationParameters* hps);
  ~HcalDDDSimulationConstants();

  const HcalSimulationParameters* hcalsimpar() const { return hspar_; }

private:
  const HcalSimulationParameters* hspar_;
};

#endif
