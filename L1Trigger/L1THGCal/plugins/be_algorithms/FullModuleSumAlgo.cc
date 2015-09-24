#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace HGCalTriggerBackend;

class FullModuleSumAlgo : public Algorithm<HGCalBestChoiceCodec> 
{
    public:

        FullModuleSumAlgo(const edm::ParameterSet& conf):
            Algorithm<HGCalBestChoiceCodec>(conf),
            cluster_product( new l1t::HGCalClusterBxCollection ){}

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll,
                const std::unique_ptr<HGCalTriggerGeometryBase>& geom) override final;

        virtual void putInEvent(edm::Event& evt) override final 
        {
            evt.put(cluster_product,name());
        }

        virtual void reset() override final 
        {
            cluster_product.reset( new l1t::HGCalClusterBxCollection );
        }

    private:
        std::auto_ptr<l1t::HGCalClusterBxCollection> cluster_product;

};

/*****************************************************************/
void FullModuleSumAlgo::run(const l1t::HGCFETriggerDigiCollection& coll,
        const std::unique_ptr<HGCalTriggerGeometryBase>& geom) 
/*****************************************************************/
{
    for( const auto& digi : coll ) 
    {
        HGCalBestChoiceCodec::data_type data;
        data.reset();
        const HGCTriggerDetId& moduleId = digi.getDetId<HGCTriggerDetId>();
        digi.decode(codec_, data);

        // Sum of trigger cells inside the module
        uint32_t moduleSum = 0;
        for(const auto& value : data.payload)
        {
            moduleSum += value;
        }
        // dummy cluster without position
        // moduleId filled in place of hardware eta
        l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                moduleSum, moduleId, 0);

        cluster_product->push_back(0,cluster);
    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        FullModuleSumAlgo,
        "FullModuleSumAlgo");
