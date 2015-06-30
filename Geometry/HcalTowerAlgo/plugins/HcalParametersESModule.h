#ifndef Geometry_HcalTowerAlgo_HcalParametersESModule_H
#define Geometry_HcalTowerAlgo_HcalParametersESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <boost/shared_ptr.hpp>
 
namespace edm {
  class ConfigurationDescriptions;
}
class HcalParameters;
class HcalParametersRcd;

class  HcalParametersESModule : public edm::ESProducer
{
 public:
  HcalParametersESModule( const edm::ParameterSet & );
  ~HcalParametersESModule( void );
  
  typedef boost::shared_ptr<HcalParameters> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const HcalParametersRcd & );
};
 
#endif
