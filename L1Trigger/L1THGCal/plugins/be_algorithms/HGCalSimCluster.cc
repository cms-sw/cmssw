// HGCal Trigger 
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
//#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

// HGCalClusters and detId
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

// PF Cluster definition
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"


// Consumes
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

//
#include "DataFormats/Math/interface/LorentzVector.h"

// Print out something before crashing or throwing exceptions
#include <iostream>


/* Original Author: Andrea Carlo Marini
 * Original Dat: 23 Aug 2016
 *
 * This backend algorithm is supposed to use the sim cluster information to handle an
 * optimal clustering algorithm, benchmark the performances of the enconding ...
 */

#define DEBUG



namespace HGCalTriggerBackend{

    template<typename FECODEC, typename DATA>
        class HGCalTriggerSimCluster : public Algorithm<FECODEC>
    {
        private:
            std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product;
            // token
            edm::EDGetTokenT< std::vector<SimCluster> > sim_token;
            // handle
            edm::Handle< std::vector<SimCluster> > sim_handle;
            // add to cluster
            void addToCluster(std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >& cluster_container, uint64_t pid,int pdgid,float  energy,float  eta, float phi)
            {
            auto iterator = cluster_container.find (pid);
            if (iterator == cluster_container.end())
                {
                    //create an empty cluster
                    cluster_container[pid] = std::pair<int,l1t::HGCalCluster>(0,l1t::HGCalCluster());
                    iterator = cluster_container.find (pid);
                    iterator -> second . second . setPdgId(pdgid);
                }
            // p4 += p4' 
            math::PtEtaPhiMLorentzVectorD p4;
            p4.SetPt ( iterator -> second . second . pt()   ) ;
            p4.SetEta( iterator -> second . second . eta()  ) ;
            p4.SetPhi( iterator -> second . second . phi()  ) ;
            p4.SetM  ( iterator -> second . second . mass() ) ;
            math::PtEtaPhiMLorentzVectorD pp4; 
            float t = std::exp (- eta);
            pp4.SetPt ( energy * (1-t*t)/(1+t*t)  ) ;
            pp4.SetEta( eta ) ;
            pp4.SetPhi( phi ) ;
            pp4.SetM  (  0  ) ;
            p4 += pp4;
            iterator -> second . second . setP4(p4);
            return ;
            }

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
                // vector<SimCluster>                    "mix"                       "MergedCaloTruth"   "HLT/DIGI"
                // pf clusters cannot be safely cast to SimCluster
                sim_token = cc.consumes< std::vector< SimCluster > >(edm::InputTag("mix","MergedCaloTruth","DIGI")); 
            }

            // setProduces
            virtual void setProduces(edm::EDProducer& prod) const override final
            {
                prod.produces<l1t::HGCalClusterBxCollection>(name());
            }

            // putInEvent
            virtual void putInEvent(edm::Event& evt) override final
            {
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] start"<<endl;
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] cluster product (!=NULL) "<<cluster_product.get()<<endl;
#endif
                evt.put(std::move(cluster_product),name());
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] DONE"<<endl;
#endif
            }

            //reset
            virtual void reset() override final 
            {
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] start"<<endl;
#endif
                cluster_product.reset( new l1t::HGCalClusterBxCollection );
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] DONE"<<endl;
#endif
            }

            // run, actual algorithm
            virtual void run( const l1t::HGCFETriggerDigiCollection & coll,
                    const edm::ESHandle<HGCalTriggerGeometryBase>&geom,
		            const edm::Event&evt
                    )
            {
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::[run] Start"<<endl;
#endif
                //1. construct a cluster container that hosts the cluster per truth-particle
                std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> > cluster_container;// PID-> bx,cluster
                evt.getByToken(sim_token,sim_handle);

                if (not sim_handle.isValid()) { std::cout<<"[HGCalTriggerSimCluster]::[run]::[ERROR] PFCluster collection for HGC sim clustering not available"<<std::endl; throw 39;}

                // 1.5. pre-process the sim cluster to have easy accessible information
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::[run] processing sim clusters"<<endl;
#endif
                // I want a map cell-> [ (pid, fraction),  ... 
                std::unordered_map<uint32_t, std::vector<std::pair< uint64_t, float > > > simclusters;
                for (auto& cluster : *sim_handle)
                {
                    auto pid= cluster.particleId(); // not pdgId
                    const auto& hf = cluster.hits_and_fractions();
                    for (const auto & p : hf ) 
                    {
                        simclusters[p.first].push_back( std::pair<uint64_t, float>( pid,p.second) ) ;
                    }
                }
                
#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::[run] Run on digis"<<endl;
#endif

                //2. run on the digits,
                for( const auto& digi : coll ) 
                {
                    DATA data;
                    digi.decode(codec_,data);
                    //2.A get the trigger-cell information energy/id
                    const HGCTriggerDetId& moduleId = digi.getDetId<HGCTriggerDetId>(); // this is a module Det Id

                    // there is a loss of generality here, due to the restriction imposed by the data formats
                    // it will work if inside a module there is a data.payload with an ordered list of all the energies
                    // one may think to add on top of it a wrapper if this stop to be the case for some of the data classes
                    for(const auto& triggercell : data.payload)
                    { 
                            if(triggercell.hwPt()<=0) continue;

                            const HGCalDetId tcellId(triggercell.detId());
                            //uint32_t digiEnergy = data.payload; i
                            auto digiEnergy=triggercell.p4().E();  
                            double eta=triggercell.p4().Eta();
                            double phi=triggercell.p4().Phi();
                            //2.B get the HGCAL-base-cell associated to it / geometry
                            //const auto& tc=geom->triggerCells()[ tcellId() ] ;//HGCalTriggerGeometry::TriggerCell&
                            //for(const auto& cell : tc.components() )  // HGcell -- unsigned
                            for(const auto& cell : geom->getCellsFromTriggerCell( tcellId()) )  // HGCcell -- unsigned
                            {
                                HGCalDetId cellId(cell);
                                //2.C get the particleId and energy fractions
                                const auto& particles =  simclusters[cellId]; // vector pid fractions
                                for ( const auto& p: particles ) 
                                {
                                    const auto & pid= p.first;
                                    const auto & fraction=p.second;
                                    auto energy = fraction*digiEnergy;
                                    //2.D add to the corresponding cluster
                                    //void addToCluster(std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >& cluster_container, uint64_t pid,int pdgid,float & energy,float & eta, float &phi)
                                    //addToCluster(cluster_container, pid, 0 energy,ETA/PHI?  ) ;
                                    addToCluster(cluster_container, pid, 0,energy,eta,phi  ) ; // how do I get eta, phi w/o the hgcal geometry?
                                }
                            }
                    } //end of for-loop
                }

#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::[run] Push clusters in cluster products"<<endl;
#endif

                //3. Push the clusters in the cluster_product
                //uint32_t clusterEnergyHw=0;
                //uint32_t clusterEtaHw = 0 ;//tcellId();
                //const GlobalPoint& tcellPosition = geom->getTriggerCellPosition( tcellId());

                // construct it from *both* physical and integer values
                //l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), 
                //        clusterEnergyHw, clusterEtaHw, 0);
                //
                for (auto&  p : cluster_container) 
                {
                    //std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >
                    #ifdef DEBUG
                    cout<<"[HGCalTriggerSimCluster]::[run] Cluster: pid="<< p.first<<endl;
                    cout<<"[HGCalTriggerSimCluster]::[run] Cluster: ----------- l1t pt"<< p.second.second.pt() <<endl;
                    cout<<"[HGCalTriggerSimCluster]::[run] Cluster: ----------- l1t eta"<< p.second.second.eta() <<endl;
                    cout<<"[HGCalTriggerSimCluster]::[run] Cluster: ----------- l1t phi"<< p.second.second.phi() <<endl;
                    #endif
                    cluster_product->push_back(p.second.first,p.second.second); // bx,cluster
                }

#ifdef DEBUG
                cout<<"[HGCalTriggerSimCluster]::["<<__FUNCTION__<<"] cluster product (!=NULL) "<<cluster_product.get()<<endl;
                cout<<"[HGCalTriggerSimCluster]::[run] END"<<endl;
#endif
            } // end run


    }; // end class

}// namespace


// define plugins, template needs to be spelled out here, in order to allow the compiler to compile, and the factory to be populated
//typedef HGCalTriggerBackend::HGCalTriggerSimCluster<HGCalBestChoiceCodec,HGCalBestChoiceDataPayload> HGCalTriggerSimClusterBestChoice;
//DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimClusterBestChoice,"HGCalTriggerSimClusterBestChoice");
typedef HGCalTriggerBackend::HGCalTriggerSimCluster<HGCalTriggerCellBestChoiceCodec,HGCalTriggerCellBestChoiceDataPayload> HGCalTriggerSimClusterBestChoice;
DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimClusterBestChoice,"HGCalTriggerSimClusterBestChoice");
//DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimCluster,"HGCalTriggerSimCluster");

// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 
