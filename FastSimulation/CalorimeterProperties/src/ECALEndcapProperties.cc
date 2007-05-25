#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/ECALEndcapProperties.h"


ECALEndcapProperties::ECALEndcapProperties(const edm::ParameterSet& fastDet)
{
  lightColl = fastDet.getParameter<double>("ECALEndcap_LightCollection");
}
