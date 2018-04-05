#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodec.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace HGCalTriggerBackend;

class RandomClusterAlgo : public Algorithm<HGCal64BitRandomCodec> {
public:
  
  RandomClusterAlgo(const edm::ParameterSet& conf,edm::ConsumesCollector &cc):
    Algorithm<HGCal64BitRandomCodec>(conf,cc),
    cluster_product_( new l1t::HGCalClusterBxCollection ){
  }

  void setProduces(edm::stream::EDProducer<>& prod) const final {
    prod.produces<l1t::HGCalClusterBxCollection>(name());
  }

  void run(const l1t::HGCFETriggerDigiCollection& coll,
		  const edm::EventSetup& es,
		  edm::Event&evt
		   ) final;

  void putInEvent(edm::Event& evt) final {
    evt.put(std::move(cluster_product_),name());
  }

  void reset() final {
    cluster_product_.reset( new l1t::HGCalClusterBxCollection );
  }
  
private:
  std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;

};

void RandomClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll,
			    const edm::EventSetup& es,
			    edm::Event&evt
			    ) {
  for( const auto& digi : coll ) {
    HGCal64BitRandomCodec::data_type my_data;
    digi.decode(codec_,my_data);

    unsigned word1 =  my_data.payload        & 0xffff;
    unsigned word2 = (my_data.payload >> 16) & 0xffff;
    unsigned word3 = (my_data.payload >> 32) & 0xffff;
    unsigned word4 = (my_data.payload >> 48) & 0xffff;

    l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                               word1, word2, word3^word4 );
    
    cluster_product_->push_back(0,cluster);
  }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
                  RandomClusterAlgo,
                  "RandomClusterAlgo");
