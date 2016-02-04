/** \file
 *
 *  $Date: 2011/04/16 10:04:23 $
 *  $Revision: 1.8 $
 *  \author Nicola Amapane 11/08
 */

#include "MagneticField/GeomBuilder/plugins/AutoMagneticFieldESProducer.h"

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

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
  nominalCurrents = pset.getUntrackedParameter<vector<int> >("nominalCurrents"); 
  maps = pset.getUntrackedParameter<vector<string> >("mapLabels");

  if (maps.size()==0 || (maps.size() != nominalCurrents.size())) {
    throw cms::Exception("InvalidParameter") << "Invalid values for parameters \"nominalCurrents\" and \"maps\"";
  }
}


AutoMagneticFieldESProducer::~AutoMagneticFieldESProducer()
{
}


std::auto_ptr<MagneticField>
AutoMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
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


  edm::ESHandle<MagneticField> map;
  
  iRecord.get(model,map);

  MagneticField* result = map.product()->clone();

  std::auto_ptr<MagneticField> s(result);
  
  return s;
}


std::string AutoMagneticFieldESProducer::closerModel(float current) {
  int i=0;
  for(;i<(int)maps.size()-1;i++) {
    if(2*current < nominalCurrents[i]+nominalCurrents[i+1] )
      return maps[i];
  }
  return  maps[i];
}



#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoMagneticFieldESProducer);


