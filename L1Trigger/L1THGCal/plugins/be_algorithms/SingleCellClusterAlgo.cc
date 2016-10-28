#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

using namespace HGCalTriggerBackend;

class SingleCellClusterAlgo : public Algorithm<HGCalTriggerCellBestChoiceCodec> 
{
    public:

        SingleCellClusterAlgo(const edm::ParameterSet& conf):
            Algorithm<HGCalTriggerCellBestChoiceCodec>(conf),
            cluster_product_( new l1t::HGCalTriggerCellBxCollection ){}

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>(name());
        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll) override final;

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

};

/*****************************************************************/
void SingleCellClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll) 
/*****************************************************************/
{
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
                cluster_product_->push_back(0,triggercell);
            }
        }

    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgo,
        "SingleCellClusterAlgo");
