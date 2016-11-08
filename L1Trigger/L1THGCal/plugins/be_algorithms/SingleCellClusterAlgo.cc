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
            calibrate_(conf){}

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
        HGCalTriggerCellCalibration calibrate_;    
};

/*****************************************************************/
void SingleCellClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) 
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
                HGCalDetId detid(triggercell.detId());
                std::cout << "" << std::endl;
                std::cout << " ------ Before calibration:" << std::endl;
                std::cout << "trgCell hwPt,Pt,Eta,Phi,E = (" <<  triggercell.hwPt()
                          << ", " << triggercell.p4().Pt() 
                          << ", " << triggercell.p4().Eta()
                          << ", " << triggercell.p4().Phi() 
                          << ", " << triggercell.p4().E() 
                          << " ) layer = " << detid.layer() << std::endl;
                l1t::HGCalTriggerCell triggercellcopy(triggercell);
                calibrate_.calibTrgCell(triggercellcopy, es);     
                //std::cout << " inside the singleCellClusterAlgo: "<< triggercell.p4().Pt() << ", " <<  triggercellcopy.p4().Pt() << std::endl;
                HGCalDetId detid_copy(triggercellcopy.detId());
                std::cout << " ------ After calibration:" << std::endl;
                std::cout << "trgCell hwPt,Pt,Eta,Phi,E = (" <<  triggercellcopy.hwPt()
                          << ", " << triggercellcopy.p4().Pt() 
                          << ", " << triggercellcopy.p4().Eta()
                          << ", " << triggercellcopy.p4().Phi() 
                          << ", " << triggercellcopy.p4().E() 
                          << " ) layer = " << detid_copy.layer() << std::endl;
                std::cout << ""<< std::endl;
                cluster_product_->push_back(0,triggercellcopy);
            }
        }
    }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        SingleCellClusterAlgoThreshold,
        "SingleCellClusterAlgoThreshold");
