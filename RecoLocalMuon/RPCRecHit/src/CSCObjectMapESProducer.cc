// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCObjectMap.h"

class CSCObjectMapESProducer : public edm::ESProducer {
public:
  CSCObjectMapESProducer(const edm::ParameterSet&) : rpcGeomToken_(setWhatProduced(this).consumes<RPCGeometry>()) {}

  std::unique_ptr<CSCObjectMap> produce(MuonGeometryRecord const& record) {
    return std::make_unique<CSCObjectMap>(record.get(rpcGeomToken_));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("cscObjectMapESProducer", desc);
  }

private:
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
};

//define this as a plug-in
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(CSCObjectMapESProducer);
