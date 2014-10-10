/** \file
 *
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
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"

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

  MagFieldConfig conf(pset, debug);

  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.get("magfield",cpv );
  MagGeoBuilderFromDDD builder(conf.version,
			       conf.geometryVersion,
			       debug);

  // Set scaling factors
  if (conf.keys.size() != 0) {
    builder.setScaling(conf.keys, conf.values);
  }
  
  // Set specification for the grid tables to be used.
  if (conf.gridFiles.size()!=0) {
    builder.setGridFiles(conf.gridFiles);
  }
  
  builder.build(*cpv);

  // Get slave field (from ES)
  edm::ESHandle<MagneticField> paramField;
  if (pset.getParameter<bool>("useParametrizedTrackerField")) {;
    iRecord.get(pset.getParameter<string>("paramLabel"),paramField);
  }
  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(conf.geometryVersion,builder.barrelLayers(), builder.endcapSectors(), builder.barrelVolumes(), builder.endcapVolumes(), builder.maxR(), builder.maxZ(), paramField.product(), false));
  
  return s;
}





DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer);
