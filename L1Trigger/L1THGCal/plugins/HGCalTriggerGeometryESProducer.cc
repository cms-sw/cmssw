
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class HGCalTriggerGeometryESProducer : public edm::ESProducer 
{
    public:
        HGCalTriggerGeometryESProducer(const edm::ParameterSet&);
        ~HGCalTriggerGeometryESProducer();

        typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

        ReturnType produce(const CaloGeometryRecord&);

    private:
        edm::ParameterSet geometry_config_;
        std::string geometry_name_;
};

HGCalTriggerGeometryESProducer::
HGCalTriggerGeometryESProducer(const edm::ParameterSet& iConfig):
    geometry_config_(iConfig.getParameterSet("TriggerGeometry")),
    geometry_name_(geometry_config_.getParameter<std::string>("TriggerGeometryName"))
{
    setWhatProduced(this);
}


HGCalTriggerGeometryESProducer::
~HGCalTriggerGeometryESProducer()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}

HGCalTriggerGeometryESProducer::ReturnType
HGCalTriggerGeometryESProducer::
produce(const CaloGeometryRecord& iRecord)
{
    //using namespace edm::es;
    ReturnType geometry(HGCalTriggerGeometryFactory::get()->create(geometry_name_,geometry_config_));
    geometry->reset();
    edm::ESHandle<CaloGeometry> calo_geometry;
    iRecord.get(calo_geometry);
    geometry->initialize(calo_geometry);
    return geometry;

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalTriggerGeometryESProducer);
