/** \class VolumeBasedMagneticFieldESProducerFromDB
 *
 *  Producer for the VolumeBasedMagneticField, taking all inputs (geometry, etc) from DB
 *
 */

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/MFGeometryFileRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "CondFormats/DataRecord/interface/MagFieldConfigRcd.h"

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace magneticfield;
using namespace edm;

namespace magneticfield {
  class VolumeBasedMagneticFieldESProducerFromDB : public edm::ESProducer {
  public:
    VolumeBasedMagneticFieldESProducerFromDB(const edm::ParameterSet& iConfig);
  
    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord & iRecord);

  private:
    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducerFromDB(const VolumeBasedMagneticFieldESProducerFromDB&) = delete;
    const VolumeBasedMagneticFieldESProducerFromDB& operator=(const VolumeBasedMagneticFieldESProducerFromDB&) = delete;
    std::string closerNominalLabel(float current);

    edm::ParameterSet pset;
    std::vector<int> nominalCurrents;
    std::vector<std::string> nominalLabels;

  };
}


VolumeBasedMagneticFieldESProducerFromDB::VolumeBasedMagneticFieldESProducerFromDB(const edm::ParameterSet& iConfig) : pset(iConfig)
{
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
  nominalCurrents={-1, 0,9558,14416,16819,18268,19262};
  nominalLabels  ={"3.8T","0T","2T", "3T", "3.5T", "3.8T", "4T"};
}



// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> VolumeBasedMagneticFieldESProducerFromDB::produce(const IdealMagneticFieldRecord & iRecord)
{

  bool debug = pset.getUntrackedParameter<bool>("debugBuilder", false);

  // Get value of the current from condition DB
  float current = pset.getParameter<int>("valueOverride");
  string message;
  if (current < 0) {
    ESHandle<RunInfo> rInfo;
    iRecord.getRecord<RunInfoRcd>().get(rInfo);
    current = rInfo->m_avg_current;
    message = " (from RunInfo DB)";
  } else {
    message = " (from valueOverride card)";
  }
  string configLabel  = closerNominalLabel(current);

  // Get configuration
  ESHandle<MagFieldConfig> confESH;
  iRecord.getRecord<MagFieldConfigRcd>().get(configLabel, confESH);
  const MagFieldConfig* conf = &*confESH;

  edm::LogInfo("MagneticField|AutoMagneticField") << "Current: " << current << message << "; using map configuration with label: " << configLabel << endl
						  << "Version: " << conf->version 
						  << " geometryVersion: " << conf->geometryVersion
						  << " slaveFieldVersion: " << conf->slaveFieldVersion;

  // Get the parametrized field
  std::unique_ptr<MagneticField> paramField = ParametrizedMagneticFieldFactory::get(conf->slaveFieldVersion, conf->slaveFieldParameters);
  

  if (conf->version == "parametrizedMagneticField") {
    // The map consist of only the parametrization in this case
    return paramField;
  } else {
    // Full VolumeBased map + parametrization
    MagGeoBuilderFromDDD builder(conf->version,
				 conf->geometryVersion,
				 debug);

    // Set scaling factors
    if (!conf->keys.empty()) {
      builder.setScaling(conf->keys, conf->values);
    }
  
    // Set specification for the grid tables to be used.
    if (!conf->gridFiles.empty()) {
      builder.setGridFiles(conf->gridFiles);
    }

    // Build the geomeytry (DDDCompactView) from the DB blob
    // (code taken from GeometryReaders/XMLIdealGeometryESSource/src/XMLIdealMagneticFieldGeometryESProducer.cc) 
    edm::ESTransientHandle<FileBlob> gdd;
    iRecord.getRecord<MFGeometryFileRcd>().get( std::to_string(conf->geometryVersion), gdd );

    auto cpv = std::make_unique<DDCompactView>(DDName("cmsMagneticField:MAGF"));
    DDLParser parser(*cpv);
    parser.getDDLSAX2FileHandler()->setUserNS(true);
    parser.clearFiles();
    std::unique_ptr<std::vector<unsigned char> > tb = (*gdd).getUncompressedBlob();
    parser.parse(*tb, tb->size());
    cpv->lockdown();
    
    builder.build(*cpv);

    // Build the VB map. Ownership of the parametrization is transferred to it
    return std::make_unique<VolumeBasedMagneticField>(conf->geometryVersion,builder.barrelLayers(), builder.endcapSectors(), builder.barrelVolumes(), builder.endcapVolumes(), builder.maxR(), builder.maxZ(), paramField.release(), true);
  }
}


std::string VolumeBasedMagneticFieldESProducerFromDB::closerNominalLabel(float current) {

  int i=0;
  for(;i<(int)nominalLabels.size()-1;i++) {
    if(2*current < nominalCurrents[i]+nominalCurrents[i+1] )
      return nominalLabels[i];
  }
  return nominalLabels[i];
}


DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducerFromDB);
