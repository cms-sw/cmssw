/** \file
 *
 *  $Date: 2013/05/21 12:57:27 $
 *  $Revision: 1.9 $
 */

#include "MagneticField/GeomBuilder/plugins/VolumeBasedMagneticFieldESProducer.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include <string>
#include <vector>
#include <iostream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace magneticfield;

VolumeBasedMagneticFieldESProducer::VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig) : pset(iConfig)
{
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}

// ------------ method called to produce the data  ------------
std::auto_ptr<MagneticField> VolumeBasedMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  bool debug = pset.getUntrackedParameter<bool>("debugBuilder", false);
  if (debug) {
    cout << "VolumeBasedMagneticFieldESProducer::produce() " << pset.getParameter<std::string>("version") << endl;
  }
  
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.get("magfield",cpv );
  MagGeoBuilderFromDDD builder(pset.getParameter<std::string>("version"),
			       pset.getParameter<int>("geometryVersion"),
			       debug);

  // Get scaling factors
  vector<int> keys = pset.getParameter<vector<int> >("scalingVolumes");
  vector<double> values = pset.getParameter<vector<double> >("scalingFactors");

  if (keys.size() != 0) {
    builder.setScaling(keys, values);
  }
  
  // Get specification for the grid tables to be used.
  typedef vector<edm::ParameterSet> VPSet;
  {
    VPSet fileSpec = pset.getParameter<VPSet>("gridFiles");
    if (fileSpec.size()!=0) {
      auto_ptr<TableFileMap> gridFiles(new TableFileMap);
      for(VPSet::const_iterator rule = fileSpec.begin(); rule != fileSpec.end(); ++rule){
	string s_volumes = rule->getParameter<string>("volumes");
	string s_sectors = rule->getParameter<string>("sectors"); // 0 means all volumes
	int master       = rule->getParameter<int>("master");
	string path      = rule->getParameter<string>("path");

	vector<unsigned> volumes = expandList(s_volumes);
	vector<unsigned> sectors = expandList(s_sectors);

	if (debug) {
	  cout << s_volumes << endl;
	  copy(volumes.begin(), volumes.end(), ostream_iterator<unsigned>(cout, " "));
	  cout << endl;
	  cout << s_sectors << endl;
	  copy(sectors.begin(), sectors.end(), ostream_iterator<unsigned>(cout, " "));
	  cout << endl;
	}
	
	for (vector<unsigned>::iterator i = volumes.begin(); i!=volumes.end(); ++i){
	  for (vector<unsigned>::iterator j = sectors.begin(); j!=sectors.end(); ++j){
	    unsigned vpacked = (*i)*100+(*j);
	    if (gridFiles->find(vpacked)==gridFiles->end()) {
	      (*gridFiles)[vpacked] = make_pair(path, master);
	    } else {
	      edm::LogError("BADconfiguration") << "ERROR: VolumeBasedMagneticFieldESProducer: malformed gridFiles config parameter";
	      abort();
	    }
	  }
	}
      }
      builder.setGridFiles(gridFiles); // gridFiles passed by reference, which is required to exist until  
    }
  }
  
  builder.build(*cpv);


  // Get slave field
  edm::ESHandle<MagneticField> paramField;
  if (pset.getParameter<bool>("useParametrizedTrackerField")) {;
    iRecord.get(pset.getParameter<string>("paramLabel"),paramField);
  }
  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(pset,builder.barrelLayers(), builder.endcapSectors(), builder.barrelVolumes(), builder.endcapVolumes(), builder.maxR(), builder.maxZ(), paramField.product(), false));
  return s;
}

vector<unsigned> VolumeBasedMagneticFieldESProducer::expandList(const string& list) {
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
      edm::LogError("BADconfiguration") << "VolumeBasedMagneticFieldESProducer: malformed configuration" << list << endl;
      abort();
    }
    for (unsigned k = start; k<=end; ++k){
      values.push_back(k);
    }
  }
  return values;
}






DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer);
