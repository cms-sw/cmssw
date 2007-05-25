#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"



HCALProperties::HCALProperties(const edm::ParameterSet& fastDet) : CalorimeterProperties()
{
  hOPi = fastDet.getParameter<double>("HCAL_PiOverE");
  spotFrac= fastDet.getParameter<double>("HCAL_Sampling");
 
}
