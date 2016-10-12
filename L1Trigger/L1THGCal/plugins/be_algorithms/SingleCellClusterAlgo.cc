#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace HGCalTriggerBackend;

class SingleCellClusterAlgo : public Algorithm<HGCalTriggerCellBestChoiceCodec> 
{
    public:

        SingleCellClusterAlgo(const edm::ParameterSet& conf):
            Algorithm<HGCalTriggerCellBestChoiceCodec>(conf),
            cluster_product_( new l1t::HGCalClusterBxCollection ){}

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll) override final;

        virtual void putInEvent(edm::Event& evt) override final 
        {
            evt.put(std::move(cluster_product_),name());
        }

        virtual void reset() override final 
        {
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );
        }

    private:
        std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;

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
        //const HGCalDetId& moduleId = digi.getDetId<HGCalDetId>();
        digi.decode(codec_, data);
        int i = 0;
        for(const auto& triggercell : data.payload)
        {
            if(triggercell.hwPt()>0)
            {
                HGCalDetId detid(triggercell.detId());
                //GlobalPoint point = geometry_->getModulePosition(moduleId);
                //math::PtEtaPhiMLorentzVector p4((double)value/cosh(.eta()), point.eta(), point.phi(), 0.);
                // index in module stored as hwEta
                l1t::HGCalCluster cluster( 
                        triggercell.p4(),
                        triggercell.hwPt(), i, 0);
                //cluster.setP4(triggercell.p4);
                cluster.setModule(module_id.wafer());
                cluster.setLayer(detid.layer());
                cluster.setSubDet(detid.subdetId());
                cluster_product_->push_back(0,cluster);
            }
            i++;
        }

    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgo,
        "SingleCellClusterAlgo");
