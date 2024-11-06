#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TTree.h>
#include <typeinfo>
#include <chrono>
#include <map>
#include <fstream>
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
#include "Phase2RawToDigi/Phase2DAQProducer/interface/DTCAssembly.h"
#include <unordered_map>

#include "Phase2RawToDigi/Phase2DAQProducer/interface/Cluster.h"

class DAQTester : public edm::one::EDProducer<> {
public:
    explicit DAQTester(const edm::ParameterSet&);
    ~DAQTester() override;

private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> ClusterCollectionToken_;
    const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
    
    // const unsigned int Positive_Modules[108];
    // const unsigned int Negative_Modules[108];

    uint32_t assignNumber(const uint32_t& N);
    void processClusters(TrackerGeometry::ModuleType moduleType,
                                           const Phase2TrackerCluster1DCollectionNew::DetSet& DetectorClusterCollection,
                                           unsigned int dtc_id, unsigned int slink_id, unsigned int slink_id_within,
                                           DTCAssembly& dtcAssembly);
};

DAQTester::DAQTester(const edm::ParameterSet& iConfig)
    : ClusterCollectionToken_(consumes<Phase2TrackerCluster1DCollectionNew>(iConfig.getParameter<edm::InputTag>("Phase2Clusters"))),
      cablingMapToken_(esConsumes()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>())
    //   Positive_Modules{30, 42, 54, 66, 78, 90, 102, 6, 18, /*...*/},
    //   Negative_Modules{138, 150, 162, 174, 186, 198, 210, 114, /*...*/}
{
    produces<FEDRawDataCollection>();
}

DAQTester::~DAQTester() 
{

}

void DAQTester::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    // Retrieve TrackerGeometry and TrackerTopology from EventSetup
    const TrackerGeometry& trackerGeometry = iSetup.getData(trackerGeometryToken_);
    // const TrackerTopology& trackerTopology = iSetup.getData(trackerTopologyToken_);
    
    // Retrieve the CablingMap
    const auto& cablingMap = iSetup.getData(cablingMapToken_);

    // get EventID and RunID
    unsigned int EventID = iEvent.id().event();

    DTCAssembly dtcAssembly (EventID);

    // Create FEDRawDataCollection to store the output
    auto fedRawDataCollection = std::make_unique<FEDRawDataCollection>();

    // Iterate through Cluster Collection
    for (const auto& DetectorClusterCollection : iEvent.get(ClusterCollectionToken_)) 
    {
        auto output = cablingMap.detIdToDTCELinkId(assignNumber(DetectorClusterCollection.detId()));

        // Retrieve DTC ID and GBT ID

        if (output.first == output.second) 
        {
            continue;
        }

        unsigned int dtc_id, gbt_id, slink_id, slink_id_within;
        for (auto it = output.first; it != output.second; ++it) 
        {
            DTCELinkId dtcELinkId = it->second;
            dtc_id = dtcELinkId.dtc_id();
            gbt_id = dtcELinkId.gbtlink_id();
        }

        slink_id = std::div(gbt_id, 18).quot;
        slink_id_within = std::div(gbt_id, 18).rem;
    

        // Example usage of TrackerGeometry
        DetId detId = DetectorClusterCollection.detId();
        const GeomDetUnit* detUnit = trackerGeometry.idToDetUnit(detId);

        if (detUnit)
        {
            
            bool isPSModulePixel = trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
            bool isPSModuleStrip = trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS;
            bool is2SModule      = trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2SS;

            if (isPSModuleStrip)
            {
                processClusters(TrackerGeometry::ModuleType::Ph2PSS, DetectorClusterCollection, dtc_id, slink_id, slink_id_within, dtcAssembly);
            }
            else if (isPSModulePixel)
            {
                processClusters(TrackerGeometry::ModuleType::Ph2PSP, DetectorClusterCollection, dtc_id, slink_id, slink_id_within, dtcAssembly);
            }
            else if (is2SModule)
            {
                processClusters(TrackerGeometry::ModuleType::Ph2SS, DetectorClusterCollection, dtc_id, slink_id, slink_id_within, dtcAssembly);
            }

        }
    }

    int i = 81;

    // for (int i = 1; i == 1; ++i)
    // {

        DTCUnit& DTC_0 = dtcAssembly.GetDTCUnit(i); // dtc unit
        
        std::vector<std::vector<Cluster>> SLink0 = DTC_0.getClustersOnSLink(0);
        std::vector<std::vector<Cluster>> SLink1 = DTC_0.getClustersOnSLink(1);
        std::vector<std::vector<Cluster>> SLink2 = DTC_0.getClustersOnSLink(2);
        std::vector<std::vector<Cluster>> SLink3 = DTC_0.getClustersOnSLink(3);
        
        DTC_0.convertToRawData(0);
        DTC_0.convertToRawData(1);
        DTC_0.convertToRawData(2);
        DTC_0.convertToRawData(3);

        // std::cout << (int)DTC_0.getDTCType() << std::endl;
        
        fedRawDataCollection.get()->FEDData( 0 + 4 * (i - 1) + 0 ) = DTC_0.GetSLink(0);
        fedRawDataCollection.get()->FEDData( 1 + 4 * (i - 1) + 0 ) = DTC_0.GetSLink(1);
        fedRawDataCollection.get()->FEDData( 2 + 4 * (i - 1) + 0 ) = DTC_0.GetSLink(2);
        fedRawDataCollection.get()->FEDData( 3 + 4 * (i - 1) + 0 ) = DTC_0.GetSLink(3);

    // }

    iEvent.put(std::move(fedRawDataCollection));

    // for (auto& cluster : SLink0.at(0))
    // {
    //     std::cout << cluster.getX() << ", " << cluster.getZ() << ", " << cluster.getChipId() << ", " << cluster.getSclusterAddress() << ", " << cluster.getWidth() << std::endl;
    // }
}

void DAQTester::processClusters(TrackerGeometry::ModuleType moduleType,
                                           const Phase2TrackerCluster1DCollectionNew::DetSet& DetectorClusterCollection,
                                           unsigned int dtc_id, unsigned int slink_id, unsigned int slink_id_within,
                                           DTCAssembly& dtcAssembly)
{
    for (const auto& cluster : DetectorClusterCollection)
    {
        DTCUnit& assignedDtcUnit = dtcAssembly.GetDTCUnit(dtc_id);

        unsigned int z = cluster.column();
        double x = cluster.center();

        // info that goes into the DAQ payload        
        unsigned int width = cluster.size();
        unsigned int chipId = 0;
        unsigned int sclusterAddress = 0;
        unsigned int mipbit = 0;
        unsigned int cicId = 0;
        // ------------------------------ //

        if (moduleType == TrackerGeometry::ModuleType::Ph2PSP) 
        {
            cicId = (z > 15);
        } 
        
        else if (moduleType == TrackerGeometry::ModuleType::Ph2PSS || moduleType == TrackerGeometry::ModuleType::Ph2SS) 
        {
            cicId = z;
        }

        if (moduleType == TrackerGeometry::ModuleType::Ph2PSP || moduleType == TrackerGeometry::ModuleType::Ph2PSS) 
        {
            chipId = std::div(x * 2.0, 120).quot;
            sclusterAddress = std::div(x * 2.0, 120).rem;
            mipbit = cluster.threshold();
        }

        else if (moduleType == TrackerGeometry::ModuleType::Ph2SS) 
        {
            chipId = std::div(x * 2.0, 254).quot;
            sclusterAddress = std::div(x * 2.0, 254).rem;
        }

        Cluster newCluster(z, x, width, chipId, sclusterAddress, mipbit, cicId, moduleType);
        assignedDtcUnit.getClustersOnSLink(slink_id).at(slink_id_within).push_back(newCluster);
        
    }
}

uint32_t DAQTester::assignNumber(const uint32_t& N) 
{
    int R = N % 4;
    if (R == 1 || R == 2) {
        return N - R;
    } else {
        return -1;
    }
}

DEFINE_FWK_MODULE(DAQTester);