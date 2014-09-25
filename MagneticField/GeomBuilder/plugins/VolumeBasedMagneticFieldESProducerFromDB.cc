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
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

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
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"

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
  
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord & iRecord);

  private:
    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducerFromDB(const VolumeBasedMagneticFieldESProducerFromDB&);
    const VolumeBasedMagneticFieldESProducerFromDB& operator=(const VolumeBasedMagneticFieldESProducerFromDB&);

    std::string closerModel(float current);

    edm::ParameterSet pset;
  };
}


VolumeBasedMagneticFieldESProducerFromDB::VolumeBasedMagneticFieldESProducerFromDB(const edm::ParameterSet& iConfig) : pset(iConfig)
{
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}



// ------------ method called to produce the data  ------------
std::auto_ptr<MagneticField> VolumeBasedMagneticFieldESProducerFromDB::produce(const IdealMagneticFieldRecord & iRecord)
{

  bool debug = pset.getUntrackedParameter<bool>("debugBuilder", false);
  if (debug) {
    cout << "VolumeBasedMagneticFieldESProducerFromDB::produce() " << pset.getParameter<std::string>("version") << endl;
  }

  // Value of the current from condition DB
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
  string model  = closerModel(current);
  edm::LogInfo("MagneticField|AutoMagneticField") << "Current: " << current << message << "; using map with label: " << model;


  //TAKE THIS FROM DB, BASED ON CURRENT
  MagFieldConfig* conf = new MagFieldConfig(pset, debug);

  MagGeoBuilderFromDDD builder(conf->version,
			       conf->geometryVersion,
			       debug);

  // Set scaling factors
  if (conf->keys.size() != 0) {
    builder.setScaling(conf->keys, conf->values);
  }
  
  // Set specification for the grid tables to be used.
  if (conf->gridFiles.size()!=0) {
    builder.setGridFiles(conf->gridFiles);
  }
  

  // build the DDDCompactView from the DB blob
  // (code taken from GeometryReaders/XMLIdealGeometryESSource/src/XMLIdealMagneticFieldGeometryESProducer.cc) 
  edm::ESTransientHandle<FileBlob> gdd;
  iRecord.getRecord<GeometryFileRcd>().get( boost::lexical_cast<string>(conf->geometryVersion), gdd );

  DDName ddName("cmsMagneticField:MAGF");
  DDLogicalPart rootNode(ddName);
  DDRootDef::instance().set(rootNode);
  std::auto_ptr<DDCompactView> cpv(new DDCompactView(rootNode));
  DDLParser parser(*cpv);
  parser.getDDLSAX2FileHandler()->setUserNS(true);
  parser.clearFiles();
  std::unique_ptr<std::vector<unsigned char> > tb = (*gdd).getUncompressedBlob();
  parser.parse(*tb, tb->size());
  cpv->lockdown();

  builder.build(*cpv);


  // Get slave field (FIXME build directly)
  edm::ESHandle<MagneticField> paramField;
  if (conf->slaveFieldVersion!="") {
    iRecord.get(conf->slaveFieldVersion,paramField);
  }
  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(pset,builder.barrelLayers(), builder.endcapSectors(), builder.barrelVolumes(), builder.endcapVolumes(), builder.maxR(), builder.maxZ(), paramField.product(), false));

  delete conf;
  return s;
}


//FIXME
 std::string VolumeBasedMagneticFieldESProducerFromDB::closerModel(float current) {
//   int i=0;
//   for(;i<(int)maps.size()-1;i++) {
//     if(2*current < nominalCurrents[i]+nominalCurrents[i+1] )
//       return maps[i];
//   }
//   return  maps[i];
   return "090322_3_8t";
 }


DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducerFromDB);
