/** \file
 *
 *  \author N. Amapane - Torino
 */

#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <vector>
#include <memory>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace magneticfield;


MagFieldConfig::MagFieldConfig(const edm::ParameterSet& pset, bool debug) {
  
  version = pset.getParameter<std::string>("version");
  geometryVersion = pset.getParameter<int>("geometryVersion");


  // Get specification for the grid tables to be used.
  typedef vector<edm::ParameterSet> VPSet;
  
  VPSet fileSpec = pset.getParameter<VPSet>("gridFiles");
  if (!fileSpec.empty()) {
    for(VPSet::const_iterator rule = fileSpec.begin(); rule != fileSpec.end(); ++rule){
      string s_volumes = rule->getParameter<string>("volumes");
      string s_sectors = rule->getParameter<string>("sectors"); // 0 means all volumes
      int master       = rule->getParameter<int>("master");
      string path      = rule->getParameter<string>("path");

      vector<unsigned> volumes = expandList(s_volumes);
      vector<unsigned> sectors = expandList(s_sectors);

      if (debug) {
	cout << "Volumes: " << s_volumes <<" Sectors: " << s_sectors 
	     << " Master: " << master << " Path:   " << path << endl;
	cout << " Expanded volumes: ";
	copy(volumes.begin(), volumes.end(), ostream_iterator<unsigned>(cout, " "));
	cout << endl;
	cout << " Expanded sectors: ";
	copy(sectors.begin(), sectors.end(), ostream_iterator<unsigned>(cout, " "));
	cout << endl;
      }
	
      for (vector<unsigned>::iterator i = volumes.begin(); i!=volumes.end(); ++i){
	for (vector<unsigned>::iterator j = sectors.begin(); j!=sectors.end(); ++j){
	  unsigned vpacked = (*i)*100+(*j);
	  if (gridFiles.find(vpacked)==gridFiles.end()) {
	    gridFiles[vpacked] = make_pair(path, master);
	  } else {
	    throw cms::Exception("ConfigurationError") << "VolumeBasedMagneticFieldESProducer: malformed gridFiles config parameter" << endl;
	  }
	}
      }
    }
  }

  // Get scaling factors
  keys = pset.getParameter<vector<int> >("scalingVolumes");
  values = pset.getParameter<vector<double> >("scalingFactors");


  // Slave field label. Either a label of an existing map (legacy support), or the 
  // type of parametrization to be constructed with the "paramData" parameters.
  slaveFieldVersion = pset.getParameter<string>("paramLabel");
  // Check for compatibility with older configurations
  if (pset.existsAs<vector<double> >("paramData")) {
    slaveFieldParameters = pset.getParameter<vector<double> >("paramData");
  }

}

vector<unsigned> MagFieldConfig::expandList(const string& list) {
  typedef vector<string> vstring;
  vector<unsigned> values;
  vstring v1;
  boost::split(v1, list, boost::is_any_of(","));
  for (vstring::const_iterator i= v1.begin(); i!=v1.end(); ++i){
    vstring v2;
    boost::split(v2, *i, boost::is_any_of("-"));	
    unsigned start = boost::lexical_cast<unsigned>(v2.front());
    unsigned end   = boost::lexical_cast<unsigned>(v2.back());
    if ((v2.size()>2) || (start>end)) {
      throw cms::Exception("ConfigurationError") << "VolumeBasedMagneticFieldESProducerFromDB: malformed configuration" << list << endl;
    }
    for (unsigned k = start; k<=end; ++k){
      values.push_back(k);
    }
  }
  return values;
}
