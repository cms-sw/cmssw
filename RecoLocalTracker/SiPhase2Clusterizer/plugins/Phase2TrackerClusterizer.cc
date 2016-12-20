#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoLocalTracker/SiPhase2Clusterizer/interface/Phase2TrackerClusterizerAlgorithm.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <memory>

class Phase2TrackerClusterizer : public edm::stream::EDProducer<> {

    public:
        explicit Phase2TrackerClusterizer(const edm::ParameterSet& conf);
        virtual ~Phase2TrackerClusterizer();
        virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

    private:
        std::unique_ptr< Phase2TrackerClusterizerAlgorithm > clusterizer_;
        edm::EDGetTokenT< edm::DetSetVector< Phase2TrackerDigi > > token_;

};


    /*
     * Initialise the producer
     */ 

    Phase2TrackerClusterizer::Phase2TrackerClusterizer(edm::ParameterSet const& conf) :
        clusterizer_(new Phase2TrackerClusterizerAlgorithm(conf.getParameter< unsigned int >("maxClusterSize"), conf.getParameter< unsigned int >("maxNumberClusters"))),
        token_(consumes< edm::DetSetVector< Phase2TrackerDigi > >(conf.getParameter<edm::InputTag>("src"))) {
            produces< Phase2TrackerCluster1DCollectionNew >(); 
    }

    Phase2TrackerClusterizer::~Phase2TrackerClusterizer() { }

    /*
     * Clusterize the events
     */

    void Phase2TrackerClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {

        // Get the Digis
        edm::Handle< edm::DetSetVector< Phase2TrackerDigi > > digis;
        event.getByToken(token_, digis);
        
        // Get the geometry
        edm::ESHandle< TrackerGeometry > geomHandle;
        eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
        const TrackerGeometry* tkGeom(&(*geomHandle)); 

        // Global container for the clusters of each modules
        auto outputClusters = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

        // Go over all the modules
        for (auto DSViter : *digis) {
            DetId detId(DSViter.detId());

            // Geometry
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            const PixelGeomDetUnit* pixDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
            if (!pixDet) assert(0);

            // Container for the clusters that will be produced for this modules
            Phase2TrackerCluster1DCollectionNew::FastFiller clusters(*outputClusters, DSViter.detId());

            // Setup the clusterizer algorithm for this detector (see ClusterizerAlgorithm for more details)
            clusterizer_->setup(pixDet);

            // Pass the list of Digis to the main algorithm
            // This function will store the clusters in the previously created container
            clusterizer_->clusterizeDetUnit(DSViter, clusters);

            if (clusters.empty()) clusters.abort();
        }

        // Add the data to the output
        outputClusters->shrink_to_fit();
        event.put(std::move(outputClusters));
    }

DEFINE_FWK_MODULE(Phase2TrackerClusterizer);
