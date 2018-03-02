#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap2DImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap3DImpl.h"


using namespace HGCalTriggerBackend;


template<typename FECODEC, typename DATA>
class HGCTowerAlgo : public Algorithm<FECODEC> 
{
    public:
        using Algorithm<FECODEC>::name;
    
    protected:
        using Algorithm<FECODEC>::codec_;


    public:

        HGCTowerAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& cc) :
        Algorithm<FECODEC>(conf, cc),
        trgcell_product_( new l1t::HGCalTriggerCellBxCollection ),
        towermap_product_( new l1t::HGCalTowerMapBxCollection ),
        tower_product_( new l1t::HGCalTowerBxCollection ),
        calibration_( conf.getParameterSet("calib_parameters") ),
        towermap2D_( conf.getParameterSet("towermap_parameters") ),
        towermap3D_( )
        {
        }
	
    
        void setProduces(edm::stream::EDProducer<>& prod) const final
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>( "calibratedTriggerCellsTower" );            
            prod.produces<l1t::HGCalTowerMapBxCollection>( "towerMap" );
            prod.produces<l1t::HGCalTowerBxCollection>( "tower" );   
        }
          
            
        void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, edm::Event&evt ) final;


        void putInEvent(edm::Event& evt) final
        {

        }


        void reset() final
        {
            trgcell_product_.reset( new l1t::HGCalTriggerCellBxCollection );    
            towermap_product_.reset( new l1t::HGCalTowerMapBxCollection );
            tower_product_.reset( new l1t::HGCalTowerBxCollection );
        }

    
    private:

        /* pointers to collections of trigger-cells, towerMaps and towers */
        std::unique_ptr<l1t::HGCalTriggerCellBxCollection> trgcell_product_;
        std::unique_ptr<l1t::HGCalTowerMapBxCollection> towermap_product_;
        std::unique_ptr<l1t::HGCalTowerBxCollection> tower_product_;
    
        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

        /* algorithms instances */
        HGCalTriggerCellCalibration calibration_;
        HGCalTowerMap2DImpl towermap2D_;
        HGCalTowerMap3DImpl towermap3D_;
};


template<typename FECODEC, typename DATA>
void HGCTowerAlgo<FECODEC,DATA>::run(const l1t::HGCFETriggerDigiCollection & coll, 
                                       const edm::EventSetup & es,
                                       edm::Event & evt ) 
{
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);    
    calibration_.eventSetup(es);
    towermap2D_.eventSetup(es);

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
                trgcell_product_->push_back( 0, calibratedtriggercell );
            }           
        
        }
    
    }

    /* orphan handles to the collections of trigger-cells, towermaps and towers */
    edm::OrphanHandle<l1t::HGCalTriggerCellBxCollection> triggerCellsHandle;
    edm::OrphanHandle<l1t::HGCalTowerMapBxCollection> towerMapsHandle;
    edm::OrphanHandle<l1t::HGCalTowerBxCollection> towersHandle;
 
    /* retrieve the orphan handle to the trigger-cells collection and put the collection in the event */
    triggerCellsHandle = evt.put( std::move( trgcell_product_ ), "calibratedTriggerCellsTower");

    /* create a persistent vector of pointers to the trigger-cells */
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
    for( unsigned i = 0; i < triggerCellsHandle->size(); ++i ) {
        edm::Ptr<l1t::HGCalTriggerCell> ptr(triggerCellsHandle,i);
        triggerCellsPtrs.push_back(ptr);
    }    

    /* call to towerMap2D clustering */
    towermap2D_.buildTowerMap2D( triggerCellsPtrs, *towermap_product_);

    /* retrieve the orphan handle to the towermaps collection and put the collection in the event */
    towerMapsHandle = evt.put( std::move( towermap_product_ ), "towerMap");

    /* create a persistent vector of pointers to the towerMaps */
    std::vector<edm::Ptr<l1t::HGCalTowerMap>> towerMapsPtrs;
    for( unsigned i = 0; i < towerMapsHandle->size(); ++i ) {
        edm::Ptr<l1t::HGCalTowerMap> ptr(towerMapsHandle,i);
        towerMapsPtrs.push_back(ptr);
    }

    /* call to towerMap3D clustering */
    towermap3D_.buildTowerMap3D( towerMapsPtrs, *tower_product_);

    /* retrieve the orphan handle to the tower collection and put the collection in the event */
    towersHandle = evt.put( std::move( tower_product_ ), "tower");

}

typedef HGCTowerAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> HGCTowerAlgoBestChoice;
typedef HGCTowerAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> HGCTowerAlgoThreshold;


DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCTowerAlgoBestChoice,
        "HGCTowerAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCTowerAlgoThreshold,
        "HGCTowerAlgoThreshold");
