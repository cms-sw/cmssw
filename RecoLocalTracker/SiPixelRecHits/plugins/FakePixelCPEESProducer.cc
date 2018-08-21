#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelFakeCPE.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include <string>
#include <memory>


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

namespace {

class  FakePixelCPEESProducer final : public edm::ESProducer{
public:

  FakePixelCPEESProducer(const edm::ParameterSet & p) {
    std::string myname = p.getParameter<std::string>("ComponentName");
    setWhatProduced(this,myname);
  }

  ~FakePixelCPEESProducer() = default; 

  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &) {
     return std::make_unique<PixelFakeCPE>();
  }

private:

};

}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(FakePixelCPEESProducer);

