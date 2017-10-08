#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace HGCalTriggerBackend;

template<typename FECODEC, typename DATA>
class FullModuleSumAlgo : public Algorithm<FECODEC> 
{
    public:

        using Algorithm<FECODEC>::name;

    protected:
        using Algorithm<FECODEC>::codec_;

    public:    

        FullModuleSumAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& cc):
            Algorithm<FECODEC>(conf,cc),
            cluster_product_( new l1t::HGCalClusterBxCollection )
	{}

        void setProduces(edm::stream::EDProducer<>& prod) const final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
        }

        void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, edm::Event&evt ) final;
        void putInEvent(edm::Event& evt) final 
        {
            evt.put(std::move(cluster_product_),name());
        }

        void reset() final 
        {
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );
        }

    private:
        std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;

};

/*****************************************************************/
template<typename FECODEC, typename DATA>
void FullModuleSumAlgo<FECODEC,DATA>::run(const l1t::HGCFETriggerDigiCollection& coll, 
			const edm::EventSetup& es,
			edm::Event&evt
			) 
/*****************************************************************/
{
    for( const auto& digi : coll ) 
    {
        DATA data;
        data.reset();
        const HGCalDetId& moduleId = digi.getDetId<HGCalDetId>();
        digi.decode(codec_, data);

        // Sum of trigger cells inside the module
        uint32_t moduleSum = 0;
        for(const auto& triggercell : data.payload)
        {
            moduleSum += triggercell.hwPt();
        }
        // dummy cluster without position
        // moduleId filled in place of hardware eta
        l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                moduleSum, 0, 0);
        cluster.setModule(moduleId.wafer());
        cluster_product_->push_back(0,cluster);
    }
}

typedef FullModuleSumAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> FullModuleSumAlgoBestChoice;
typedef FullModuleSumAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> FullModuleSumAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
                  FullModuleSumAlgoBestChoice,
                  "FullModuleSumAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory,
                  FullModuleSumAlgoThreshold,
                  "FullModuleSumAlgoThreshold");
