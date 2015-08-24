// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/RPCRecHit/src/DTObjectMap.h"

class DTObjectMapESProducer : public edm::ESProducer {
public:
  DTObjectMapESProducer(const edm::ParameterSet&) {
    setWhatProduced(this);
  }

  ~DTObjectMapESProducer() {
  }

  boost::shared_ptr<DTObjectMap> produce(MuonGeometryRecord const& record) {
    return boost::make_shared<DTObjectMap>(record);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("dtObjectMapESProducer", desc);
  }

};

//define this as a plug-in
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(DTObjectMapESProducer);
