#ifndef Geometry_HcalCommonData_CaloSimulationConstants_h
#define Geometry_HcalCommonData_CaloSimulationConstants_h

/** \class CaloSimulationConstants
 *
 * this class reads the constant section of
 * the xml-files related to calorimeter utility for Calo simulation
 *  
 * \author Sunanda Banerjee, FNAL <sunanda.banerjee@cern.ch>
 *
 */

#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"

class CaloSimulationConstants {
public:
  CaloSimulationConstants(const CaloSimulationParameters* cps);
  ~CaloSimulationConstants();

  const CaloSimulationParameters* caloSimPar() const { return calospar_; }

private:
  const CaloSimulationParameters* calospar_;
};

#endif
