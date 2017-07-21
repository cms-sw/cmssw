#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"    

using namespace HGCalTriggerBackend;


template<typename FECODEC, typename DATA>
class HGCClusterAlgo : public Algorithm<FECODEC> 
{
    public:
        using Algorithm<FECODEC>::name;
    
    protected:
        using Algorithm<FECODEC>::codec_;

    private:
        enum ClusterType{
            dRC2d,
            NNC2d
        };
    
    public:

        HGCClusterAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& cc) :
        Algorithm<FECODEC>(conf, cc),
        trgcell_product_( new l1t::HGCalTriggerCellBxCollection ),
        cluster_product_( new l1t::HGCalClusterBxCollection ),
        multicluster_product_( new l1t::HGCalMulticlusterBxCollection ),
        calibration_( conf.getParameterSet("calib_parameters") ),
        clustering_( conf.getParameterSet("C2d_parameters") ),
        multiclustering_( conf.getParameterSet("C3d_parameters" ) )
        {
            clustering_threshold_silicon_ = conf.getParameterSet("C2d_parameters").getParameter<double>("clustering_threshold_silicon");
            clustering_threshold_scintillator_ = conf.getParameterSet("C2d_parameters").getParameter<double>("clustering_threshold_scintillator");
            std::string type(conf.getParameterSet("C2d_parameters").getParameter<std::string>("clusterType"));
            if(type=="dRC2d"){
                clusteringAlgorithmType_ = dRC2d;
            }else if(type=="NNC2d"){
                clusteringAlgorithmType_ = NNC2d;
            }else {
                edm::LogWarning("ParameterError") << "Unknown clustering type '" << type
                    << "'. Using nearest neighbor NNC2d instead.\n";
                clusteringAlgorithmType_ = NNC2d;
            }

        }
    
        virtual void setProduces(edm::stream::EDProducer<>& prod) const override final
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>( "calibratedTriggerCells" );            
            prod.produces<l1t::HGCalClusterBxCollection>( "cluster2D" );
            prod.produces<l1t::HGCalMulticlusterBxCollection>( "cluster3D" );   
        }
            
        virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, edm::Event&evt ) override final;


        virtual void putInEvent(edm::Event& evt) override final 
        {

        }
    

        virtual void reset() override final 
        {
            trgcell_product_.reset( new l1t::HGCalTriggerCellBxCollection );            
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );
            multicluster_product_.reset( new l1t::HGCalMulticlusterBxCollection );
        }

    
    private:
    
        /* pointers to collections of trigger-cells, clusters and multiclusters */
        std::unique_ptr<l1t::HGCalTriggerCellBxCollection> trgcell_product_;
        std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;
        std::unique_ptr<l1t::HGCalMulticlusterBxCollection> multicluster_product_;
    
        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

        /* algorithms instances */
        HGCalTriggerCellCalibration calibration_;
        HGCalClusteringImpl clustering_;
        HGCalMulticlusteringImpl multiclustering_;

        /* algorithm type */
        ClusterType clusteringAlgorithmType_;
        double clustering_threshold_silicon_;
        double clustering_threshold_scintillator_;
};


template<typename FECODEC, typename DATA>
void HGCClusterAlgo<FECODEC,DATA>::run(const l1t::HGCFETriggerDigiCollection & coll, 
                                       const edm::EventSetup & es,
                                       edm::Event & evt ) 
{
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);

    for( const auto& digi : coll ){
        
        HGCalDetId module_id( digi.id() );
        
        DATA data;
        data.reset();
        digi.decode(codec_, data);
        
        for(const auto& triggercell : data.payload)
        {
            
            if( triggercell.hwPt() > 0 )
            {
                l1t::HGCalTriggerCell calibratedtriggercell( triggercell );
                calibration_.calibrateInGeV( calibratedtriggercell); 
                double clustering_threshold = (triggercell.subdetId()==HGCHEB ? clustering_threshold_scintillator_ : clustering_threshold_silicon_);
                if(calibratedtriggercell.mipPt()<clustering_threshold) continue;
                trgcell_product_->push_back( 0, calibratedtriggercell );
            }           
        
        }
    
    }

    /* orphan handles to the collections of trigger-cells, clusters and multiclusters */
    edm::OrphanHandle<l1t::HGCalTriggerCellBxCollection> triggerCellsHandle;
    edm::OrphanHandle<l1t::HGCalClusterBxCollection> clustersHandle;
    edm::OrphanHandle<l1t::HGCalMulticlusterBxCollection> multiclustersHandle;
    
    /* retrieve the orphan handle to the trigger-cells collection and put the collection in the event */
    triggerCellsHandle = evt.put( std::move( trgcell_product_ ), "calibratedTriggerCells");

    /* create a persistent vector of pointers to the trigger-cells */
    edm::PtrVector<l1t::HGCalTriggerCell> triggerCellsPtrs;
    for( unsigned i = 0; i < triggerCellsHandle->size(); ++i ) {
        edm::Ptr<l1t::HGCalTriggerCell> ptr(triggerCellsHandle,i);
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
        default:
            // Should not happen, clustering type checked in constructor
            break;
    }

    /* retrieve the orphan handle to the clusters collection and put the collection in the event */
    clustersHandle = evt.put( std::move( cluster_product_ ), "cluster2D");

    /* create a persistent vector of pointers to the trigger-cells */
    edm::PtrVector<l1t::HGCalCluster> clustersPtrs;
    for( unsigned i = 0; i < clustersHandle->size(); ++i ) {
        edm::Ptr<l1t::HGCalCluster> ptr(clustersHandle,i);
        clustersPtrs.push_back(ptr);
    }
    
    /* call to multiclustering */
    multiclustering_.clusterize( clustersPtrs, *multicluster_product_ );

    /* retrieve the orphan handle to the multiclusters collection and put the collection in the event */
    multiclustersHandle = evt.put( std::move( multicluster_product_ ), "cluster3D");

}

typedef HGCClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> HGCClusterAlgoBestChoice;
typedef HGCClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> HGCClusterAlgoThreshold;


DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCClusterAlgoBestChoice,
        "HGCClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCClusterAlgoThreshold,
        "HGCClusterAlgoThreshold");
