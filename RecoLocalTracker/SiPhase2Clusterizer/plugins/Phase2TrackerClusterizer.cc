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

#include "Phase2TrackerClusterizerAlgorithm.h"
#include "Phase2TrackerClusterizerSequentialAlgorithm.h"

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
        auto outputClustersNew = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

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

            Phase2TrackerCluster1DCollectionNew::FastFiller clustersNew(*outputClustersNew, DSViter.detId());
            Phase2TrackerClusterizerSequentialAlgorithm	algo;
	    algo.clusterizeDetUnit(DSViter, clustersNew);
            if (clustersNew.empty()) clustersNew.abort();

            if (clusters.size() != clustersNew.size()) {
              std::cout << "SIZEs " << int(detId) << ' ' << clusters.size() << ' ' << clustersNew.size() << std::endl;
              for (auto const & cl : clusters) std::cout << cl.size() << ' ' << cl.threshold() << ' ' << cl.firstRow() << ' ' << cl.column() << std::endl;
              std::cout << "new " << std::endl;
              for (auto const & cl : clustersNew) std::cout << cl.size() << ' ' << cl.threshold() << ' ' << cl.firstRow() << ' ' << cl.column() << std::endl;
            }
        }
        
        std::cout << "SIZEs " << outputClusters->dataSize() << ' ' << outputClustersNew->dataSize() << std::endl;
        // assert(outputClusters->dataSize()==outputClustersNew->dataSize());
        for (auto i=0U;i<outputClusters->dataSize(); ++i) {
          assert(outputClusters->data()[i].size() == outputClustersNew->data()[i].size());
       	  assert(outputClusters->data()[i].threshold() == outputClustersNew->data()[i].threshold());
          assert(outputClusters->data()[i].firstRow() == outputClustersNew->data()[i].firstRow());
          assert(outputClusters->data()[i].column() == outputClustersNew->data()[i].column());
        }


        // Add the data to the output
        outputClusters->shrink_to_fit();
        event.put(std::move(outputClusters));
    }

DEFINE_FWK_MODULE(Phase2TrackerClusterizer);
