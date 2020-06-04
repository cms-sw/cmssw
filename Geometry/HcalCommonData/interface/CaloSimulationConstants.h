#ifndef Geometry_HcalCommonData_CaloDDDSimulationConstants_h
#define Geometry_HcalCommonData_CaloDDDSimulationConstants_h

/** \class CaloDDDSimulationConstants
 *
 * this class reads the constant section of
 * the xml-files related to calorimeter utility for Calo simulation
 *  
 * \author Sunanda Banerjee, FNAL <sunanda.banerjee@cern.ch>
 *
 */

#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"

class CaloDDDSimulationConstants {
public:
  CaloDDDSimulationConstants(const CaloSimulationParameters* cps);
  ~CaloDDDSimulationConstants();

  const CaloSimulationParameters* caloSimPar() const { return calospar_; }

private:
  const CaloSimulationParameters* calospar_;
};

#endif
