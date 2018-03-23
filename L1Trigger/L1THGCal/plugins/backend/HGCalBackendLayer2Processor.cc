#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringImpl.h"    


class HGCalBackendLayer2Processor : public HGCalBackendLayer2ProcessorBase 
{
    public:
      HGCalBackendLayer2Processor(const edm::ParameterSet& conf)  : 
		HGCalBackendLayer2ProcessorBase(conf),
		multiclustering_( conf.getParameterSet("C3d_parameters" ) )
      {
      }
    
      void setProduces3D(edm::stream::EDProducer<>& prod) const final
      {
            prod.produces<l1t::HGCalMulticlusterBxCollection>( "cluster3D" );   
      }
            
      void run3D(const edm::Handle<l1t::HGCalClusterBxCollection> handleColl, 
                                       const edm::EventSetup & es,
                                       edm::Event & evt ) 
      {
        es.get<CaloGeometryRecord>().get("", triggerGeometry_);
        multiclustering_.eventSetup(es);

        /* orphan handles to the collections of trigger-cells, clusters and multiclusters */
        edm::OrphanHandle<l1t::HGCalMulticlusterBxCollection> multiclustersHandle;
    
        /* create a persistent vector of pointers to the trigger-cells */
        std::vector<edm::Ptr<l1t::HGCalCluster>> clustersPtrs;
        for( unsigned i = 0; i < handleColl->size(); ++i ) {
          edm::Ptr<l1t::HGCalCluster> ptr(handleColl,i);
          clustersPtrs.push_back(ptr);
        }
    
        /* call to multiclustering and compute shower shape*/
        switch(multiclusteringAlgoType_){
          case dRC3d : 
            multiclustering_.clusterizeDR( clustersPtrs, *multicluster_product_, *triggerGeometry_);
            break;
          case DBSCANC3d:
            multiclustering_.clusterizeDBSCAN( clustersPtrs, *multicluster_product_, *triggerGeometry_);
            break;
          default:
            // Should not happen, clustering type checked in constructor
            break;
        }

        /* retrieve the orphan handle to the multiclusters collection and put the collection in the event */
        multiclustersHandle = evt.put( std::move( multicluster_product_ ), "cluster3D");
      }

      void putInEvent3D(edm::Event& evt) final 
      {
        //evt.put(std::move(multicluster_product_), "cluster3D"); 
      }
    

      void reset3D() final 
      {
            multicluster_product_.reset( new l1t::HGCalMulticlusterBxCollection );
      }

    
    private:
    
       /* pointers to collections of trigger-cells, clusters and multiclusters */
       std::unique_ptr<l1t::HGCalMulticlusterBxCollection> multicluster_product_;
    
       edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

       /* algorithms instances */
       HGCalMulticlusteringImpl multiclustering_;

       /* algorithm type */
       ClusterType clusteringAlgorithmType_;
       double triggercell_threshold_silicon_;
       double triggercell_threshold_scintillator_;
       MulticlusterType multiclusteringAlgoType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer2Factory, 
        HGCalBackendLayer2Processor,
        "HGCalBackendLayer2Processor");

