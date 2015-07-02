#ifndef Geometry_HcalTowerAlgo_HcalParametersESModule_H
#define Geometry_HcalTowerAlgo_HcalParametersESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <boost/shared_ptr.hpp>
 
namespace edm {
  class ConfigurationDescriptions;
}
class PHcalParameters;
class PHcalParametersRcd;

class  HcalParametersESModule : public edm::ESProducer
{
 public:
  HcalParametersESModule( const edm::ParameterSet & );
  ~HcalParametersESModule( void );
  
  typedef boost::shared_ptr<PHcalParameters> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const PHcalParametersRcd & );
};
 
#endif
