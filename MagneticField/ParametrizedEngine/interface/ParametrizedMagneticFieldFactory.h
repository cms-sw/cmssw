#ifndef ParametrizedMagneticFieldFactory_h
#define ParametrizedMagneticFieldFactory_h

/** \class ParametrizedMagneticFieldFactory
 *
 *  Create a parametrized field map with the specified configuration.
 *
 *  \author N. Amapane - Torino
 */


#include <MagneticField/Engine/interface/MagneticField.h>
#include <vector>
#include <string>
#include <memory>

namespace edm{
  class ParameterSet;
}

namespace magneticfield{
  class ParametrizedMagneticFieldProducer;
  class AutoParametrizedMagneticFieldProducer;
  class VolumeBasedMagneticFieldESProducerFromDB;
}

class ParametrizedMagneticFieldFactory {
 public:
  /// Constructor
  ParametrizedMagneticFieldFactory();
  
 private:
  friend class magneticfield::ParametrizedMagneticFieldProducer;
  friend class magneticfield::AutoParametrizedMagneticFieldProducer;
  friend class magneticfield::VolumeBasedMagneticFieldESProducerFromDB;

  // Get map configured from pset (deprecated)
  std::unique_ptr<MagneticField>
  static get(std::string version, const edm::ParameterSet& parameters);
  
  // Get map configured from type name and numerical parameters
  std::unique_ptr<MagneticField>
  static get(std::string version, std::vector<double> parameters);

};
#endif

