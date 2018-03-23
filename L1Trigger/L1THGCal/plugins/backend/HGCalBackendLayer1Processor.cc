#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringImpl.h"


class HGCalBackendLayer1Processor : public HGCalBackendLayer1ProcessorBase 
{
    
    public:

       HGCalBackendLayer1Processor(const edm::ParameterSet& conf)  : 
		HGCalBackendLayer1ProcessorBase(conf),
		HGCalEESensitive_( conf.getParameter<std::string>("HGCalEESensitive_tag") ),
                HGCalHESiliconSensitive_( conf.getParameter<std::string>("HGCalHESiliconSensitive_tag") ),
		clustering_( conf.getParameterSet("C2d_parameters") )
        {
            std::string typeCluster(conf.getParameterSet("C2d_parameters").getParameter<std::string>("clusterType"));
            if(typeCluster=="dRC2d"){
                clusteringAlgorithmType_ = dRC2d;
            }else if(typeCluster=="NNC2d"){
                clusteringAlgorithmType_ = NNC2d;
            }else if(typeCluster=="dRNNC2d"){
                clusteringAlgorithmType_ = dRNNC2d;
            }else {
                throw cms::Exception("HGCTriggerParameterError")
                    << "Unknown clustering type '" << typeCluster;
            }

        }
    
        void setProduces2D(edm::stream::EDProducer<>& prod) const final
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>( "calibratedTriggerCells" );            
            prod.produces<l1t::HGCalClusterBxCollection>( "cluster2D" );
        }
            
        void putInEvent2D(edm::Event& evt) final 
        {
            evt.put(std::move(cluster_product_), "cluster2D");
        }
    

        void reset2D() final 
        {
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );
        }
   
 
        void run2D(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle, 
                                       const edm::EventSetup & es,
                                       edm::Event & evt ) 
        {
          es.get<CaloGeometryRecord>().get("", triggerGeometry_);
          clustering_.eventSetup(es);

          /* create a persistent vector of pointers to the trigger-cells */
          std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
          for( unsigned i = 0; i < collHandle->size(); ++i ) {
            edm::Ptr<l1t::HGCalTriggerCell> ptr(collHandle,i);
            triggerCellsPtrs.push_back(ptr);
          }
    
          /* call to C2d clustering */
          switch(clusteringAlgorithmType_){
            case dRC2d : 
              clustering_.clusterizeDR( triggerCellsPtrs, *cluster_product_);
              break;
            case NNC2d:
              clustering_.clusterizeNN( triggerCellsPtrs, *cluster_product_, *triggerGeometry_ );
              break;
            case dRNNC2d:
              clustering_.clusterizeDRNN( triggerCellsPtrs, *cluster_product_, *triggerGeometry_ );
              break;
            default:
              // Should not happen, clustering type checked in constructor
              break;
          }
        }  
    
    private:
        enum ClusterType{
            dRC2d,
            NNC2d,
            dRNNC2d
        };
        
	/* pointers to collections of trigger-cells, clusters and multiclusters */
        std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;
    
        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

        /* algorithms instances */
        HGCalClusteringImpl clustering_;

        /* algorithm type */
        ClusterType clusteringAlgorithmType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer1Factory, 
        HGCalBackendLayer1Processor,
        "HGCalBackendLayer1Processor");
