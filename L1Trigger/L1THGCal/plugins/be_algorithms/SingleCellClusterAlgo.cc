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
        SingleCellClusterAlgo(const edm::ParameterSet& conf,edm::ConsumesCollector &cc):
            Algorithm<FECODEC>(conf,cc),
            cluster_product_( new l1t::HGCalTriggerCellBxCollection ),
            HGCalEESensitive_(conf.getParameter<std::string>("HGCalEESensitive_tag")),
            HGCalHESiliconSensitive_(conf.getParameter<std::string>("HGCalHESiliconSensitive_tag")),
            calibration_(conf.getParameterSet("calib_parameters")){}

        typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

        virtual void setProduces(edm::stream::EDProducer<>& prod) const override final 
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>(name());
        }
    
        virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es,
		         edm::Event&evt
			) override final
        {
            es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
            es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);
            
            for( const auto& digi : coll ) 
            {
                HGCalDetId module_id(digi.id());
                DATA data;
                data.reset();
                digi.decode(codec_, data);
                for(const auto& triggercell : data.payload)
                {
                    if(triggercell.hwPt()>0)
                    {
                        
                        HGCalDetId detid(triggercell.detId());
                        l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                        calibration_.calibrateInGeV(calibratedtriggercell);     
                        cluster_product_->push_back(0,calibratedtriggercell);
                    }
                }
            }
        }
 
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
        std::string HGCalEESensitive_;
        std::string HGCalHESiliconSensitive_;

        edm::ESHandle<HGCalTopology> hgceeTopoHandle_;
        edm::ESHandle<HGCalTopology> hgchefTopoHandle_;
        HGCalTriggerCellCalibration calibration_;    

};

typedef SingleCellClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> SingleCellClusterAlgoBestChoice;
typedef SingleCellClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> SingleCellClusterAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoBestChoice,
        "SingleCellClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoThreshold,
        "SingleCellClusterAlgoThreshold");
