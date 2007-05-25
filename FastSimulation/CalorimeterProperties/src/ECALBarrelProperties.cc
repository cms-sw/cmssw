#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALBarrelProperties.h"

ECALBarrelProperties::ECALBarrelProperties(const edm::ParameterSet& fastDet)
{
  lightColl =  fastDet.getParameter<double>("ECALBarrel_LightCollection");  
}
