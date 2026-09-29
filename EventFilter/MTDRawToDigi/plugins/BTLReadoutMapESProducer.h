#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/MTDObjects/interface/BTLElectronicsId.h"
#include "CondFormats/MTDObjects/interface/BTLReadoutMap.h"
#include "CondFormats/DataRecord/interface/BTLReadoutMapRcd.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

class BTLReadoutMapESProducer : public edm::ESProducer {
public:
  BTLReadoutMapESProducer(const edm::ParameterSet&);
  ~BTLReadoutMapESProducer() override;

  std::unique_ptr<BTLReadoutMap> produce(const BTLReadoutMapRcd&);

private:
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
};
