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

namespace edm {class ParameterSet;}

namespace magneticfield {
  typedef std::map<int, std::pair<std::string, int> > TableFileMap;
}


class MagFieldConfig {
public:

  MagFieldConfig(){}

  /// Constructor
  MagFieldConfig(const edm::ParameterSet& pset, bool debug=false);
  
  // Operations
public: // FIXME
  std::vector<unsigned> expandList(const std::string& list);

  //
  int geometryVersion;

  //
  std::string version;  

  //
  magneticfield::TableFileMap gridFiles;

  //scaling factors
  std::vector<int> keys;
  std::vector<double> values;
  
  std::string slaveFieldVersion;
  std::vector<double> slaveFieldParameters;

  COND_SERIALIZABLE;
};
#endif


