#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "EventFilter/Phase2TrackerRawToDigi/interface/DTCAssembly.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Cluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"

class ClusterToRawProducer : public edm::one::EDProducer<> {
public:
    explicit ClusterToRawProducer(const edm::ParameterSet&);
    ~ClusterToRawProducer() override;

private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> ClusterCollectionToken_;
    const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;

    uint32_t assignNumber(const uint32_t& N);
    void processClusters(TrackerGeometry::ModuleType moduleType,
                                           const Phase2TrackerCluster1DCollectionNew::DetSet& detector_cluster_collection,
                                           unsigned int dtc_id, unsigned int slink_id, unsigned int slink_id_within, bool is_seed_sensor,
                                           DTCAssembly& dtcAssembly);
};

ClusterToRawProducer::ClusterToRawProducer(const edm::ParameterSet& iConfig)
    : ClusterCollectionToken_(consumes<Phase2TrackerCluster1DCollectionNew>(iConfig.getParameter<edm::InputTag>("Phase2Clusters"))),
      cablingMapToken_(esConsumes()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>())
{
    produces<FEDRawDataCollection>();
}

ClusterToRawProducer::~ClusterToRawProducer() 
{

}

void ClusterToRawProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    // Retrieve TrackerGeometry and TrackerTopology from EventSetup
    const TrackerGeometry& trackerGeometry = iSetup.getData(trackerGeometryToken_);
    const TrackerTopology& trackerTopology = iSetup.getData(trackerTopologyToken_);
    
    // Retrieve the CablingMap
    const auto& cablingMap = iSetup.getData(cablingMapToken_);

    // get EventID and RunID
    unsigned int EventID = iEvent.id().event();

    DTCAssembly dtcAssembly (EventID);

    // Create FEDRawDataCollection to store the output
    auto fedRawDataCollection = std::make_unique<FEDRawDataCollection>();

    // Iterate through Cluster Collection
    for (const auto& detector_cluster_collection : iEvent.get(ClusterCollectionToken_)) 
    {
        const unsigned int detId = detector_cluster_collection.detId();
        const unsigned int cable_map_module_id = assignNumber(detId);
        auto cable_map = cablingMap.detIdToDTCELinkId(cable_map_module_id);

        // Retrieve DTC ID and GBT ID

        if (cable_map.first == cable_map.second) { continue; }

        unsigned int dtc_id, gbt_id, slink_id, slink_id_within;
        for (auto it = cable_map.first; it != cable_map.second; ++it) 
        {
            DTCELinkId dtcELinkId = it->second;
            dtc_id = dtcELinkId.dtc_id();
            gbt_id = dtcELinkId.gbtlink_id();
        }

        slink_id = std::div(gbt_id, Phase2TrackerSpecifications::MODULES_PER_SLINK).quot;
        slink_id_within = std::div(gbt_id, Phase2TrackerSpecifications::MODULES_PER_SLINK).rem;
    
        const bool is_seed_sensor = trackerTopology.isLower(detId);
        const GeomDetUnit* detUnit = trackerGeometry.idToDetUnit(detId);

        if (detUnit)
        {
            
            TrackerGeometry::ModuleType moduleType = trackerGeometry.getDetectorType(detId);

            switch (moduleType) 
            {
                case TrackerGeometry::ModuleType::Ph2PSS:
                case TrackerGeometry::ModuleType::Ph2PSP:
                case TrackerGeometry::ModuleType::Ph2SS:
                    processClusters(moduleType, detector_cluster_collection, dtc_id, slink_id, slink_id_within, is_seed_sensor, dtcAssembly);
                    break;
                default:
                    break;
            }

        }
    }

    int i = 81; // a single 2s module for unpacking tests

    DTCUnit& dtc_0 = dtcAssembly.GetDTCUnit(i); // dtc unit
    dtc_0.convertToRawData(); // convert to raw data
    for (int j = 0; j < 4; j++)
    { fedRawDataCollection.get()->FEDData( j + 4 * (i - 1) + 0 ) = dtc_0.getSLink(j); }

    iEvent.put(std::move(fedRawDataCollection));

}

void ClusterToRawProducer::processClusters(TrackerGeometry::ModuleType moduleType,
                                           const Phase2TrackerCluster1DCollectionNew::DetSet& detector_cluster_collection,
                                           unsigned int dtc_id, unsigned int slink_id, unsigned int slink_id_within, bool is_seed_sensor,
                                           DTCAssembly& dtcAssembly)
{
    for (const auto& cluster : detector_cluster_collection)
    {
        DTCUnit& assignedDtcUnit = dtcAssembly.GetDTCUnit(dtc_id);

        unsigned int z = cluster.column();
        double x = cluster.firstStrip();

        // info that goes into the DAQ payload
        unsigned int width = cluster.size();

        if (width > 8) {continue;}

        unsigned int chipId = 0;
        unsigned int sclusterAddress = 0;
        unsigned int mipbit = 0;
        unsigned int cicId = 0;
        // ------------------------------ //

        if (moduleType == TrackerGeometry::ModuleType::Ph2PSP) 
        {
            cicId = (z > Phase2TrackerSpecifications::CIC_Z_BOUNDARY) ? 1 : 0;
        }
        
        else if (moduleType == TrackerGeometry::ModuleType::Ph2PSS || moduleType == TrackerGeometry::ModuleType::Ph2SS) 
        {
            cicId = z;
        }

        if (moduleType == TrackerGeometry::ModuleType::Ph2PSP || moduleType == TrackerGeometry::ModuleType::Ph2PSS) 
        {
            chipId = std::div(x * 2.0, Phase2TrackerSpecifications::CHANNELS_PER_SSA).quot;
            sclusterAddress = std::div(x * 2.0, Phase2TrackerSpecifications::CHANNELS_PER_SSA).rem;
            mipbit = cluster.threshold();
        }

        else if (moduleType == TrackerGeometry::ModuleType::Ph2SS) 
        {
            chipId = std::div(x, Phase2TrackerSpecifications::STRIPS_PER_CBC).quot;
            sclusterAddress = std::div(x, Phase2TrackerSpecifications::STRIPS_PER_CBC).rem;
        }

        sclusterAddress = ((sclusterAddress & 0x7F) << 1) | (is_seed_sensor & 0x1);

        Cluster newCluster(z, x, width, chipId, sclusterAddress, mipbit, cicId, moduleType);
        assignedDtcUnit.getClustersOnSLink(slink_id).at(slink_id_within).push_back(newCluster);
        
    }
}

uint32_t ClusterToRawProducer::assignNumber(const uint32_t& N) 
{
    int R = N % 4;
    if (R == 1 || R == 2) {
        return N - R;
    } else {
        return -1;
    }
}

DEFINE_FWK_MODULE(ClusterToRawProducer);