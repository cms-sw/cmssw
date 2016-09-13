// HGCal Trigger 
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

// HGCalClusters and detId
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

// PF Cluster definition
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

// Consumes
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"



/* Original Author: Andrea Carlo Marini
 * Original Dat: 23 Aug 2016
 *
 * This backend algorithm is supposed to use the sim cluster information to handle an
 * optimal clustering algorithm, benchmark the performances of the enconding ...
 */


namespace HGCalTriggerBackend{
    template<typename FECODEC, typename DATA>
        class HGCalTriggerSimCluster : public Algorithm<FECODEC>
    {
        private:
            std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product;
            // token
            edm::EDGetTokenT< reco::PFClusterCollection > sim_token;
            // handle
            edm::Handle< reco::PFClusterCollection > sim_handle;

        protected:
            using Algorithm<FECODEC>::codec_;

        public:
            // Constructor
            //
            using Algorithm<FECODEC>::Algorithm; 
            using Algorithm<FECODEC>::name;
            using Algorithm<FECODEC>::run;
            using Algorithm<FECODEC>::putInEvent;
            using Algorithm<FECODEC>::setProduces;
            using Algorithm<FECODEC>::reset;

            //Consumes tokens
            HGCalTriggerSimCluster(const edm::ParameterSet& conf,edm::ConsumesCollector&cc) : Algorithm<FECODEC>(conf,cc) { 
                // I need to consumes the PF Cluster Collection with the sim clustering, TODO: make it configurable (?)
                sim_token = cc.consumes< reco::PFClusterCollection >(edm::InputTag("particleFlowClusterHGCal")); 
            }

            // setProduces
            virtual void setProduces(edm::EDProducer& prod) const override final
            {
                prod.produces<l1t::HGCalClusterBxCollection>(name());
            }

            // putInEvent
            virtual void putInEvent(edm::Event& evt) override final
            {
                evt.put(std::move(cluster_product),name());
            }

            //reset
            virtual void reset() override final 
            {
                cluster_product.reset( new l1t::HGCalClusterBxCollection );
            }

            // run, actual algorithm
            virtual void run( const l1t::HGCFETriggerDigiCollection & coll,
                    const std::unique_ptr<HGCalTriggerGeometryBase>&geom,
		            const edm::Event&evt
                    )
            {
                //1. construct a cluster container that hosts the cluster per truth-particle
                std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> > cluster_container;// PID-> bx,cluster
                evt.getByToken(sim_token,sim_handle);
                //2. run on the digits,
                for( const auto& digi : coll ) 
                {
                    DATA data;
                    const HGCTriggerDetId& tcellId = digi.getDetId<HGCTriggerDetId>();
                    digi.decode(codec_,data);
                    //2.A get the trigger-cell information energy/id
                    //uint32_t digiEnergy = data.payload; i
                    auto digiEnergy= data.energy();  // if it is implemented an energy() method, etherwise it will not compile
                    // XXX this depends on  the DATA type
                    //2.B get the HGCAL-base-cell associated to it / geometry
                    //const auto& tc=geom->triggerCells()[ tcellId() ] ;//HGCalTriggerGeometry::TriggerCell&
                    //for(const auto& cell : tc.components() )  // HGcell -- unsigned
                    for(const auto& cell : geom->getCellsFromTriggerCell( tcellId()) )  // HGCcell -- unsigned
                    {
                        HGCalDetId cellId(cell);
                        //2.C get the particleId and energy fractions
                        //2.D add to the corresponding cluster
                    }
                }

                //3. Push the clusters in the cluster_product
                uint32_t clusterEnergyHw=0;
                uint32_t clusterEtaHw = 0 ;//tcellId();
                //const GlobalPoint& tcellPosition = geom->getTriggerCellPosition( tcellId());

                // construct it from *both* physical and integer values
                l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                        clusterEnergyHw, clusterEtaHw, 0);

                cluster_product->push_back(0,cluster); // bx,cluster

            }


    };

}// namespace


// define plugins, template needs to be spelled out here, in order to allow the compiler to compile, and the factory to be populated
typedef HGCalTriggerBackend::HGCalTriggerSimCluster<HGCalBestChoiceCodec,HGCalBestChoiceDataPayload> HGCalTriggerSimClusterBestChoice;
DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimClusterBestChoice,"HGCalTriggerSimClusterBestChoice");
//DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimCluster,"HGCalTriggerSimCluster");

// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 
