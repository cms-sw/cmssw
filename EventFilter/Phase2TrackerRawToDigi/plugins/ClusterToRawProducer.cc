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
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"

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

ClusterToRawProducer::~ClusterToRawProducer() { }

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

    using namespace Phase2TrackerSpecifications;
    using namespace Phase2DAQFormatSpecification;

    for (int dtc_id = MIN_DTC_ID; dtc_id < MAX_DTC_ID + 1; dtc_id++)
    {
        for (int slink_id = MIN_SLINK_ID; slink_id < MAX_SLINK_ID + 1; slink_id++)
        {
            int index_first = slink_id * MODULES_PER_SLINK;
            int index_last = (slink_id + 1) * MODULES_PER_SLINK;

            FEDRawData slink_daq_stream;

            std::vector<Word32Bits> daq_packet;
            std::vector<Word32Bits> offset_map;
            std::vector<Word32Bits> payload;

            for (int i = 0; i < 3; ++i) { daq_packet.push_back( Word32Bits(DTC_DAQ_HEADER) ); }

            for (int module_id = index_first; module_id < index_last; module_id++) 
            {
                DTCELinkId cms_link_id = DTCELinkId(dtc_id, module_id, 0);
                try 
                {
                    
                    auto                    link_to_det_association = cablingMap.dtcELinkIdToDetId(cms_link_id);
                    const DTCELinkId&       link_id = link_to_det_association->first;
                    const DetId&            det_id = link_to_det_association->second;

                    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator 
                                    det_seed = iEvent.get(ClusterCollectionToken_).find(det_id + 1);

                    TrackerGeometry::ModuleType seed_sensor_type = trackerGeometry.getDetectorType( det_seed->detId() );


                    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator 
                                    det_corr = iEvent.get(ClusterCollectionToken_).find(det_id + 2);

                    TrackerGeometry::ModuleType corr_sensor_type = trackerGeometry.getDetectorType( det_corr->detId() );

                    for (const auto& cluster : *det_seed)
                    {
                        
                    }

                    for (const auto& cluster : *det_corr)
                    {
                        
                    }

                } 
                catch (const cms::Exception& e) 
                {
                    // exception here means that the link is not connected to a detector

                    uint32_t eventID = EventID & L1ID_MAX_VALUE;    // eventId_ (9 bits)
                    uint32_t channelErrors = 0;                     // 9 bits for errors, all set to 0
                    uint32_t numClusters = 0;                       // no clusters here.

                    // Build the channel header 
                    uint32_t header = (eventID << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS)) | 
                                    (channelErrors << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) | 
                                    (numClusters << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - NCLUSTERS_BITS));

                    // Push the header into the payload
                    payload.push_back(header);

                    continue; 
                }

                // fedRawDataCollection.get()->FEDData( slink_id + SLINKS_PER_DTC * (dtc_id - 1) + TRACKER_HEADER ) = dtc_0.getSLink(j);

            }

        }
    }

    // Iterate through Cluster Collection
    // for (const auto& detector_cluster_collection : iEvent.get(ClusterCollectionToken_)) 
    // {
    //     const unsigned int detId = detector_cluster_collection.detId();
    //     const unsigned int cable_map_module_id = assignNumber(detId);
    //     auto cable_map = cablingMap.detIdToDTCELinkId(cable_map_module_id);

    //     // Retrieve DTC ID and GBT ID

    //     if (cable_map.first == cable_map.second) { continue; }

    //     unsigned int dtc_id, gbt_id, slink_id, slink_id_within, elink_id;
    //     for (auto it = cable_map.first; it != cable_map.second; ++it) 
    //     {
    //         DTCELinkId dtcELinkId = it->second;
    //         dtc_id = dtcELinkId.dtc_id();
    //         gbt_id = dtcELinkId.gbtlink_id();
    //         elink_id = dtcELinkId.elink_id();

    //         std::cout << dtc_id << ", " << gbt_id << ", " << elink_id << std::endl;
    //     }

    //     slink_id = std::div(gbt_id, Phase2TrackerSpecifications::MODULES_PER_SLINK).quot;
    //     slink_id_within = std::div(gbt_id, Phase2TrackerSpecifications::MODULES_PER_SLINK).rem;
    
    //     const bool is_seed_sensor = trackerTopology.isLower(detId);
    //     const GeomDetUnit* detUnit = trackerGeometry.idToDetUnit(detId);

    //     if (detUnit)
    //     {
            
    //         TrackerGeometry::ModuleType moduleType = trackerGeometry.getDetectorType(detId);

    //         switch (moduleType) 
    //         {
    //             case TrackerGeometry::ModuleType::Ph2PSS:
    //             case TrackerGeometry::ModuleType::Ph2PSP:
    //             case TrackerGeometry::ModuleType::Ph2SS:
    //                 processClusters(moduleType, detector_cluster_collection, dtc_id, slink_id, slink_id_within, is_seed_sensor, dtcAssembly);
    //                 break;
    //             default:
    //                 break;
    //         }

    //     }
    // }

    // int i = 81; // a single 2s module for unpacking tests

    // DTCUnit& dtc_0 = dtcAssembly.GetDTCUnit(i); // dtc unit
    // dtc_0.convertToRawData(); // convert to raw data
    // for (int j = 0; j < 4; j++)
    // { fedRawDataCollection.get()->FEDData( j + 4 * (i - 1) + 0 ) = dtc_0.getSLink(j); }

    // iEvent.put(std::move(fedRawDataCollection));

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