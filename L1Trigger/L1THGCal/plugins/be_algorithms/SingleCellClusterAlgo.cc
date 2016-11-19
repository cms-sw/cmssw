#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

using namespace HGCalTriggerBackend;

class SingleCellClusterAlgo : public Algorithm<HGCalTriggerCellBestChoiceCodec> 
{
    public:

        SingleCellClusterAlgo(const edm::ParameterSet& conf):
            Algorithm<HGCalTriggerCellBestChoiceCodec>(conf),
            cluster_product_( new l1t::HGCalTriggerCellBxCollection ),
            HGCalEESensitive_(conf.getParameter<std::string>("HGCalEESensitive_tag")),
            HGCalHESiliconSensitive_(conf.getParameter<std::string>("HGCalHESiliconSensitive_tag")),
            calibration_(conf){}

        typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

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
        string HGCalEESensitive_;
        string HGCalHESiliconSensitive_;

        edm::ESHandle<HGCalTopology> hgceeTopoHandle_;
        edm::ESHandle<HGCalTopology> hgchefTopoHandle_;
        HGCalTriggerCellCalibration calibration_;    

};

/*****************************************************************/
void SingleCellClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) 
/*****************************************************************/
{ 
    es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
    es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);
 

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
                int subdet = detid.subdetId();
                int cellThickness = 0;

                if( subdet == HGCEE ){ 
                    cellThickness = (hgceeTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                }else if( subdet == HGCHEF ){
                    cellThickness = (hgchefTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                }else if( subdet == HGCHEB ){
                    edm::LogWarning("DataNotFound") << "ATTENTION: the BH trgCells are not yet implemented !! ";
                }
                l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                calibration_.calibrate(calibratedtriggercell, cellThickness);     
                cluster_product_->push_back(0,calibratedtriggercell);
            }
        }
    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgo,
        "SingleCellClusterAlgo");
