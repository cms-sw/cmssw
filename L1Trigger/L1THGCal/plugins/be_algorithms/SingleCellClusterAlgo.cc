#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

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
          Algorithm<FECODEC>(conf),
          cluster_product_( new l1t::HGCalTriggerCellBxCollection ){}

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalTriggerCellBxCollection>(name());
        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll) override final
        {
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
                    cluster_product_->push_back(0,triggercell);
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

};

typedef SingleCellClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> SingleCellClusterAlgoBestChoice;
typedef SingleCellClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> SingleCellClusterAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoBestChoice,
        "SingleCellClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoThreshold,
        "SingleCellClusterAlgoThreshold");
