/** \file
 *
 *  $Date: 2009/03/03 12:58:02 $
 *  $Revision: 1.4 $
 *  \author Nicola Amapane 11/08
 */

#include "MagneticField/GeomBuilder/plugins/AutoMagneticFieldESProducer.h"

#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAE85lParametrizedMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/PolyFit2DParametrizedMagneticField.h"

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "MagneticField/Engine/interface/MagneticFieldHelpers.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace magneticfield;


AutoMagneticFieldESProducer::AutoMagneticFieldESProducer(const edm::ParameterSet& iConfig) : pset(iConfig) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}


AutoMagneticFieldESProducer::~AutoMagneticFieldESProducer()
{
}


std::auto_ptr<MagneticField>
AutoMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
  MagneticField * result=0;


  int value = pset.getParameter<int>("valueOverride");

  if (value < 0) {
    ESHandle<RunInfo> rInfo;
    iRecord.getRecord<RunInfoRcd>().get(rInfo);

    float current = rInfo->m_avg_current;
    value = magneticFieldHelpers::closerNominalField(current);
    
    edm::LogInfo("MagneticField|AutoMagneticField") << "Recorded avg current: " << current << "; using map for " << value/10. << " T";
  } else {
    edm::LogInfo("MagneticField|AutoMagneticField") << "Ignoring DB current readings; using map for " << value/10. << " T";
  }

  MagneticField * paramField=0;

  string version;
  
  if (value == 0) {
    // Use UniformMagneticField instead of the inner region parametrization.
    // This trick allows to keep a VBF field to be used by propagators to check the geometry.
    paramField = new UniformMagneticField(0.);

    // Temporary hack: set 3.8T underlying VBF map so that SHP can decide if volume is iron
    //    version = "fake";
    stringstream str;
    string model = pset.getParameter<string>("model");
    str << model << "_" << "3_8t";
    version = str.str();

  } else {
    // Handle different labelling conventions
    string VBFValue;
    string OAEValue;
    
    if (value == 20) {
      VBFValue = "2t";
      OAEValue = "2_0T";
    } else if (value == 30){
      VBFValue = "3t";
      OAEValue = "3_0T";
    } else if (value == 35){
      VBFValue = "3_5t";
      OAEValue = "3_5T";
    } else if (value == 38){
      VBFValue = "3_8t";
      OAEValue = "3_8T";
    } else if (value == 40){
      VBFValue = "4t";
      OAEValue = "4_0T";
    } else {
      throw cms::Exception("InvalidParameter")<<"Invalid field value: requested : " << value << " kGauss";
    }

    stringstream str;
    string model = pset.getParameter<string>("model");
    str << model << "_" << VBFValue;
    version = str.str();

    // Build slave field
    if (pset.getParameter<bool>("useParametrizedTrackerField")) {

      string parVersion = pset.getParameter<string>("subModel");
      
      if (parVersion=="OAE_1103l_071212") { 
	// V. Karimaki's off-axis expansion fitted to v1103l TOSCA computation
	ParameterSet ppar;
	ppar.addParameter<string>("BValue", OAEValue);
	paramField =  new OAEParametrizedMagneticField(ppar);
	//      } else if (parVersion=="PolyFit2D") {
 	// V. Maroussov polynomial fit to mapping data
	// 	ParameterSet ppar;
	// 	ppar.addParameter<double>("BValue", 4.01242188708911); //FIXME
	// 	paramField = new PolyFit2DParametrizedMagneticField(ppar);
	//       }
      } else {
	throw cms::Exception("InvalidParameter")<<"Invalid parametrization version " << parVersion;
      }
    }
  }
  
  ParameterSet VBFPset;
  VBFPset.addUntrackedParameter<bool>("debugBuilder",false);
  VBFPset.addUntrackedParameter<bool>("cacheLastVolume",true);
  VBFPset.addParameter<string>("version",version);
    
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get("magfield",cpv );
  MagGeoBuilderFromDDD builder(VBFPset.getParameter<std::string>("version"),
			       VBFPset.getUntrackedParameter<bool>("debugBuilder", false));

    
  // Get scaling factors
  vector<int> keys = pset.getParameter<vector<int> >("scalingVolumes");
  vector<double> values = pset.getParameter<vector<double> >("scalingFactors");

  if (keys.size() != 0) {
    builder.setScaling(keys, values);
  }

  builder.build(*cpv);
  
  result = new VolumeBasedMagneticField(VBFPset,
					builder.barrelLayers(), 
					builder.endcapSectors(), 
					builder.barrelVolumes(), 
					builder.endcapVolumes(),
					builder.maxR(), 
					builder.maxZ(), 
					paramField, true);

  
  std::auto_ptr<MagneticField> s(result);
  return s;
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoMagneticFieldESProducer);


