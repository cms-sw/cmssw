#include "RecoLocalFastTime/FTLClusterizer/plugins/MTDCPEESProducer.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

MTDCPEESProducer::MTDCPEESProducer(const edm::ParameterSet & p) 
{
  pset_ = p;
  setWhatProduced(this,"MTDCPEBase");
}

MTDCPEESProducer::~MTDCPEESProducer() {}

std::unique_ptr<MTDClusterParameterEstimator>
MTDCPEESProducer::produce(const MTDCPERecord & iRecord)
{ 
  edm::ESHandle<MTDGeometry> pDD;
  iRecord.getRecord<MTDDigiGeometryRecord>().get( pDD );
  
  return std::make_unique<MTDCPEBase>(
                         pset_,
			 *pDD.product()
				      );
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_EVENTSETUP_MODULE(MTDCPEESProducer);
