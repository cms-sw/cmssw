#ifndef MagFieldConfig_h
#define MagFieldConfig_h

/** \class MagFieldConfig
 *
 *  No description available.
 *
 *  \author N. Amapane - Torino
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <map>

namespace edm {
  class ParameterSet;
}

namespace magneticfield {
  typedef std::map<int, std::pair<std::string, int> > TableFileMap;
}

class MagFieldConfig {
public:
  MagFieldConfig() {}

  /// Constructor
  MagFieldConfig(const edm::ParameterSet& pset, bool debug = false);

  // Operations
public:
  std::vector<unsigned> expandList(const std::string& list);

  /// Version of the geometry to be used
  int geometryVersion;

  /// Version of the data tables to be used
  std::string version;

  /// Specification of which data table is to be used for each volume
  magneticfield::TableFileMap gridFiles;

  /// Scaling factors for the field in specific volumes
  std::vector<int> keys;
  std::vector<double> values;

  /// Label or type of the tracker parametrization
  std::string slaveFieldVersion;

  /// Parameters for the tracker parametrization
  /// (not used in legacy producers where slaveFieldVersion is the label of the
  /// parametrization in the EventSetup)
  std::vector<double> slaveFieldParameters;

  COND_SERIALIZABLE;
};
#endif
