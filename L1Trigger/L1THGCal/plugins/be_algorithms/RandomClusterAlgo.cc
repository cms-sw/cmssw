#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodec.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace HGCalTriggerBackend;

class RandomClusterAlgo : public Algorithm<HGCal64BitRandomCodec> {
public:
  
  RandomClusterAlgo(const edm::ParameterSet& conf):
    Algorithm<HGCal64BitRandomCodec>(conf),
    cluster_product( new l1t::HGCalClusterBxCollection ){
  }

  virtual void setProduces(edm::EDProducer& prod) const override final {
    prod.produces<l1t::HGCalClusterBxCollection>(name());
  }

  virtual void run(const l1t::HGCFETriggerDigiCollection& coll,
                   const std::unique_ptr<HGCalTriggerGeometryBase>& geom) override final;

  virtual void putInEvent(edm::Event& evt) override final {
    evt.put(cluster_product,name());
  }

  virtual void reset() override final {
    cluster_product.reset( new l1t::HGCalClusterBxCollection );
  }
  
private:
  std::auto_ptr<l1t::HGCalClusterBxCollection> cluster_product;

};

void RandomClusterAlgo::run(const l1t::HGCFETriggerDigiCollection& coll,
                            const std::unique_ptr<HGCalTriggerGeometryBase>& geom) {
  for( const auto& digi : coll ) {
    HGCal64BitRandomCodec::data_type my_data;
    digi.decode(codec_,my_data);

    unsigned word1 =  my_data.payload        & 0xffff;
    unsigned word2 = (my_data.payload >> 16) & 0xffff;
    unsigned word3 = (my_data.payload >> 32) & 0xffff;
    unsigned word4 = (my_data.payload >> 48) & 0xffff;

    l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                               word1, word2, word3^word4 );
    
    cluster_product->push_back(0,cluster);
  }
}

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
                  RandomClusterAlgo,
                  "RandomClusterAlgo");
