#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <memory>

using namespace edm;

class  MTDTimeCalibESProducer: public edm::ESProducer
{
 public:
  MTDTimeCalibESProducer(const edm::ParameterSet & p);
  ~MTDTimeCalibESProducer() override = default; 

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  std::unique_ptr<MTDTimeCalib> produce(const MTDTimeCalibRecord &);
  
 private:
  edm::ParameterSet pset_;
};


MTDTimeCalibESProducer::MTDTimeCalibESProducer(const edm::ParameterSet & p) 
{
  pset_ = p;
  setWhatProduced(this,"MTDTimeCalib");
}

// Configuration descriptions
void
MTDTimeCalibESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("BTLTimeOffset", 0.);
  desc.add<double>("ETLTimeOffset", 0.);
  descriptions.add("MTDTimeCalibESProducer", desc);
}

std::unique_ptr<MTDTimeCalib>
MTDTimeCalibESProducer::produce(const MTDTimeCalibRecord & iRecord)
{ 
  edm::ESHandle<MTDGeometry> pDD;
  iRecord.getRecord<MTDDigiGeometryRecord>().get( pDD );
  
  edm::ESHandle<MTDTopology> pTopo;
  iRecord.getRecord<MTDTopologyRcd>().get( pTopo );
  
  return std::make_unique<MTDTimeCalib>(
					    pset_,
					    pDD.product(),
					    pTopo.product()
					    );
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_EVENTSETUP_MODULE(MTDTimeCalibESProducer);
