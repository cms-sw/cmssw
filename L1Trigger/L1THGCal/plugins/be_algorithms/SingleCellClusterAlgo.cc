#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

using namespace HGCalTriggerBackend;

template<typename FECODEC, typename DATA>
class SingleCellClusterAlgo : public Algorithm<FECODEC> 
{
    public:
        using Algorithm<FECODEC>::name;

    protected:
        using Algorithm<FECODEC>::codec_;

    public:

        SingleCellClusterAlgo(const edm::ParameterSet& conf):
            Algorithm<HGCalTriggerCellBestChoiceCodec>(conf),
            cluster_product_( new l1t::HGCalTriggerCellBxCollection ),
            calibration_(conf){}
        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>(name());
        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) override final;

        virtual void putInEvent(edm::Event& evt) override final 
        {
            evt.put(std::move(cluster_product_),name());
        }

        virtual void reset() override final 
        {
            cluster_product_.reset( new l1t::HGCalTriggerCellBxCollection );
        }

    private:
        std::unique_ptr<l1t::HGCalTriggerCellBxCollection> cluster_product_;
        edm::ESHandle<HGCalGeometry> hgceeGeoHandle;
        edm::ESHandle<HGCalGeometry> hgchefGeoHandle;
        HGCalTriggerCellCalibration calibration_;    
};

/*****************************************************************/
void SingleCellClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) 
/*****************************************************************/
{
    es.get<IdealGeometryRecord>().get("HGCalEESensitive",hgceeGeoHandle);  
    es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hgchefGeoHandle); 
    const HGCalGeometry* geom = nullptr;


    for( const auto& digi : coll ) 
    {
        HGCalDetId module_id(digi.id());
        HGCalTriggerCellBestChoiceCodec::data_type data;
        data.reset();
        digi.decode(codec_, data);
        for(const auto& triggercell : data.payload)
        {
            if(triggercell.hwPt()>0)
            {

                HGCalDetId detid(triggercell.detId());
                int subdet =  (ForwardSubdetector)detid.subdetId() - 3;
                if( subdet == 0 ){ 
                    geom = hgceeGeoHandle.product();     
                }else if( subdet == 1 ){
                    geom = hgchefGeoHandle.product();
                }else if( subdet == 2 ){
                    //std::cout << "ATTENTION: the BH trgCells are not yet implemented !! "<< std::endl;
                }
                const HGCalTopology& topo = geom->topology();
                const HGCalDDDConstants& dddConst = topo.dddConstants();
                int cellThickness = dddConst.waferTypeL((unsigned int)detid.wafer() );        

                l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                calibration_.calibrate(calibratedtriggercell, subdet, cellThickness);     
                HGCalDetId detid_copy(calibratedtriggercell.detId());
                cluster_product_->push_back(0,calibratedtriggercell);
            }
        }
    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoThreshold,
        "SingleCellClusterAlgoThreshold");
