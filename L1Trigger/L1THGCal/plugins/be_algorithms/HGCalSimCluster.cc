// HGCal Trigger 
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

// HGCalClusters and detId
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

// PF Cluster definition
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"


// Consumes
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

//
#include "DataFormats/Math/interface/LorentzVector.h"

// Energy calibration
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"



// Print out something before crashing or throwing exceptions
#include <iostream>
#include <string>


/* Original Author: Andrea Carlo Marini
 * Original Date: 23 Aug 2016
 *
 * This backend algorithm is supposed to use the sim cluster information to handle an
 * optimal clustering algorithm, benchmark the performances of the enconding ...
 * as well as performing benchmarks on cluster shapes
 */


namespace HGCalTriggerBackend{

    template<typename FECODEC, typename DATA>
        class HGCalTriggerSimCluster : public Algorithm<FECODEC>
    {
        private:
            std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;
            // handle
            edm::Handle< std::vector<SimCluster> > sim_handle_;
            // calibration handle
            edm::ESHandle<HGCalTopology> hgceeTopoHandle_;
            edm::ESHandle<HGCalTopology> hgchefTopoHandle_;        

            // variables that needs to be init, in the right order
            // energy calibration
            std::string HGCalEESensitive_;
            std::string HGCalHESiliconSensitive_;
            HGCalTriggerCellCalibration calibration_; 

            // token
            edm::EDGetTokenT< std::vector<SimCluster> > sim_token_;
            // Digis
            edm::EDGetToken inputee_, inputfh_, inputbh_;

            // add to cluster shapes, once per trigger cell
            void addToClusterShapes(std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >& cluster_container, uint64_t pid,int pdgid,float  energy,float  eta, float phi, float r=0.0){
                auto pair = cluster_container.emplace(pid, std::pair<int,l1t::HGCalCluster>(0,l1t::HGCalCluster() ) ) ;
                auto iterator = pair.first;
                iterator -> second . second . shapes().Add( energy,eta,phi,r); // last is r, for 3d clusters
            }
            // add to cluster
            void addToCluster(std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >& cluster_container, uint64_t pid,int pdgid,float  energy,float  eta, float phi)
            {

                auto pair = cluster_container.emplace(pid, std::pair<int,l1t::HGCalCluster>(0,l1t::HGCalCluster() ) ) ;
                auto iterator = pair.first;
                math::PtEtaPhiMLorentzVectorD p4;
                p4.SetPt ( iterator -> second . second . pt()   ) ;
                p4.SetEta( iterator -> second . second . eta()  ) ;
                p4.SetPhi( iterator -> second . second . phi()  ) ;
                p4.SetM  ( iterator -> second . second . mass() ) ;
                math::PtEtaPhiMLorentzVectorD pp4; 
                float t = std::exp (- eta);
                pp4.SetPt ( energy * (2*t)/(1+t*t)  ) ;
                pp4.SetEta( eta ) ;
                pp4.SetPhi( phi ) ;
                pp4.SetM  (  0  ) ;
                p4 += pp4;
                iterator -> second . second . setP4(p4);
                //iterator -> second . second . shapes().Add( energy,eta,phi,r); // last is r, for 3d clusters
                return ;
            }

            using Algorithm<FECODEC>::geometry_; 
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
            HGCalTriggerSimCluster(const edm::ParameterSet& conf,edm::ConsumesCollector&cc) : 
                            Algorithm<FECODEC>(conf,cc),
                            HGCalEESensitive_(conf.getParameter<std::string>("HGCalEESensitive_tag")),
                            HGCalHESiliconSensitive_(conf.getParameter<std::string>("HGCalHESiliconSensitive_tag")),
                            calibration_(conf.getParameterSet("calib_parameters"))
            { 
                sim_token_ = cc.consumes< std::vector< SimCluster > >(conf.getParameter<edm::InputTag>("simcollection")); 
                inputee_ = cc.consumes<edm::PCaloHitContainer>( conf.getParameter<edm::InputTag>("simhitsee"));
                inputfh_ = cc.consumes<edm::PCaloHitContainer>( conf.getParameter<edm::InputTag>("simhitsfh"));
                // inputbh_ = cc.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("g4SimHits:HGCHitsHEback")); 
                edm::LogWarning("HGCalTriggerSimCluster") <<"WARNING: BH simhits not loaded";
            }

            // setProduces
            virtual void setProduces(edm::stream::EDProducer<>& prod) const override final
            {
                prod.produces<l1t::HGCalClusterBxCollection>(name());
            }

            // putInEvent
            virtual void putInEvent(edm::Event& evt) override final
            {
                evt.put(std::move(cluster_product_),name());
            }

            //reset
            virtual void reset() override final 
            {
                cluster_product_.reset( new l1t::HGCalClusterBxCollection );
            }

            // run, actual algorithm
            virtual void run( const l1t::HGCFETriggerDigiCollection & coll,
                           const edm::EventSetup& es,
                           edm::Event&evt
                    )
            {
                //0.5. Get Digis, construct a map, detid -> energy
                
                edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
                evt.getByToken(inputee_,ee_simhits_h);

                edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
                evt.getByToken(inputfh_,fh_simhits_h);

                edm::Handle<edm::PCaloHitContainer> bh_simhits_h;
                //evt.getByToken(inputbh_,bh_simhits_h);

                
                if (not ee_simhits_h.isValid()){
                       throw cms::Exception("ContentError")<<"[HGCalTriggerSimCluster]::[run]::[ERROR] EE Digis from HGC not available"; 
                }
                if (not fh_simhits_h.isValid()){
                       throw cms::Exception("ContentError")<<"[HGCalTriggerSimCluster]::[run]::[ERROR] FH Digis from HGC not available"; 
                }

                std::unordered_map<uint64_t, double> hgc_simhit_energy;

                edm::ESHandle<HGCalTopology> topo_ee, topo_fh;
                es.get<IdealGeometryRecord>().get("HGCalEESensitive",topo_ee);
                es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",topo_fh);

                if (ee_simhits_h.isValid()) 
                {
                    int layer=0,cell=0, sec=0, subsec=0, zp=0,subdet=0;
                    ForwardSubdetector mysubdet;
                    for (const auto& simhit : *ee_simhits_h)
                    {
                        HGCalTestNumbering::unpackHexagonIndex(simhit.id(), subdet, zp, layer, sec, subsec, cell); 
                        mysubdet = (ForwardSubdetector)(subdet);
                        std::pair<int,int> recoLayerCell = topo_ee->dddConstants().simToReco(cell,layer,sec,topo_ee->detectorType());
                        cell  = recoLayerCell.first;
                        layer = recoLayerCell.second;
                        if (layer<0 || cell<0) {
                          continue;
                        }
                        unsigned recoCell = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);

                        hgc_simhit_energy[recoCell] += simhit.energy();
                    }
                }
                if (fh_simhits_h.isValid())
                {
                    int layer=0,cell=0, sec=0, subsec=0, zp=0,subdet=0;
                    ForwardSubdetector mysubdet;
                    for (const auto& simhit : *fh_simhits_h)
                    {
                        HGCalTestNumbering::unpackHexagonIndex(simhit.id(), subdet, zp, layer, sec, subsec, cell); 
                        mysubdet = (ForwardSubdetector)(subdet);
                        std::pair<int,int> recoLayerCell = topo_fh->dddConstants().simToReco(cell,layer,sec,topo_fh->detectorType());
                        cell  = recoLayerCell.first;
                        layer = recoLayerCell.second;
                        if (layer<0 || cell<0) {
                          continue;
                        }
                        unsigned recoCell = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);
                        hgc_simhit_energy[recoCell] += simhit.energy();
                    }
                }

                if (bh_simhits_h.isValid() ) /// FIXME TODO
                {
                       throw cms::Exception("Not Implemented")<<"HGCalTriggerSimCluster: BH simhits not implemnted"; 

                }
                
                //1. construct a cluster container that hosts the cluster per truth-particle
                std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> > cluster_container;// PID-> bx,cluster
                evt.getByToken(sim_token_,sim_handle_);

                if (not sim_handle_.isValid()){
                       throw cms::Exception("ContentError")<<"[HGCalTriggerSimCluster]::[run]::[ERROR] Sim Cluster collection for HGC sim clustering not available"; 
                }
                // calibration
                es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
                es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);

                // 1.5. pre-process the sim cluster to have easy accessible information
                // I want a map cell-> [ (pid, fraction),  ... 
                std::unordered_map<uint32_t, std::vector<std::pair< uint64_t, float > > > simclusters;
                for (auto& cluster : *sim_handle_)
                {
                    auto pid= cluster.particleId(); // not pdgId
                    const auto& hf = cluster.hits_and_fractions();
                    for (const auto & p : hf ) 
                    {
                        simclusters[p.first].push_back( std::pair<uint64_t, float>( pid,p.second) ) ;
                    }
                }
                
                //2. run on the digits,
                for( const auto& digi : coll ) 
                {
                    DATA data;
                    digi.decode(codec_,data);
                    //2.A get the trigger-cell information energy/id
                    //const HGCTriggerDetId& moduleId = digi.getDetId<HGCTriggerDetId>(); // this is a module Det Id

                    // there is a loss of generality here, due to the restriction imposed by the data formats
                    // it will work if inside a module there is a data.payload with an ordered list of all the energies
                    // one may think to add on top of it a wrapper if this stop to be the case for some of the data classes
                    for(const auto& triggercell : data.payload)
                    { 
                            if(triggercell.hwPt()<=0) continue;

                            const HGCalDetId tcellId(triggercell.detId());
                            // calbration

                            l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                            calibration_.calibrateInGeV(calibratedtriggercell); 
                            //uint32_t digiEnergy = data.payload; 
                            //auto digiEnergy=triggercell.p4().E();  
                            // using calibrated energy instead
                            auto calibratedDigiEnergy=calibratedtriggercell.p4().E();  
                            double eta=triggercell.p4().Eta();
                            double phi=triggercell.p4().Phi();
                            double z = triggercell.position().z(); // may be useful for cluster shapes
                            //2.B get the HGCAL-base-cell associated to it / geometry
                            
                            // normalization loop 
                            double norm=0.0;
                            map<unsigned, double> energy_for_cluster_shapes;
                            
                            for(const auto& cell : geometry_->getCellsFromTriggerCell( tcellId()) )  // HGCcell -- unsigned
                            {
                                HGCalDetId cellId(cell);

                                //2.C0 find energy of the hgc cell. default is very small value
                                double hgc_energy=1.0e-10; //average if not found / bh
                                const auto &it = hgc_simhit_energy.find(cell);
                                if (it != hgc_simhit_energy.end()) {  hgc_energy = it->second; }

                                //2.C get the particleId and energy fractions
                                const auto & iterator= simclusters.find(cellId);
                                if (iterator == simclusters.end() )  continue;
                                const auto & particles = iterator->second;
                                for ( const auto& p: particles ) 
                                {
                                    const auto & pid= p.first;
                                    const auto & fraction=p.second;
                                    norm += fraction * hgc_energy;
                                    energy_for_cluster_shapes[pid] += calibratedDigiEnergy *fraction *hgc_energy; // norm will be done later, with the above norm
                                }
                            }
                            
                            // second loop counting the energy
                            for(const auto& cell : geometry_->getCellsFromTriggerCell( tcellId()) )  // HGCcell -- unsigned
                            {
                                HGCalDetId cellId(cell);
                                double hgc_energy=1.0e-10; // 1 ->  average if not found / bh
                                const auto &it = hgc_simhit_energy.find(cell);
                                if (it != hgc_simhit_energy.end()) {  hgc_energy = it->second; }
                                //2.C get the particleId and energy fractions
                                const auto & iterator= simclusters.find(cellId);
                                if (iterator == simclusters.end() )  continue;
                                const auto & particles = iterator->second;
                                for ( const auto& p: particles ) 
                                {
                                    const auto & pid= p.first;
                                    const auto & fraction=p.second;
                                    //auto energy = fraction * calibratedDigiEnergy/norm;
                                    auto energy = fraction * hgc_energy* calibratedDigiEnergy/norm; // THIS IS WHAT I WANT
                                    //#warning FIXME_ENERGY
                                    //auto energy = fraction * hgc_energy* hgc_energy/norm;

                                    //2.D add to the corresponding cluster
                                    //eta/phi are the position of the trigger cell, to consider degradation 
                                    addToCluster(cluster_container, pid, 0,energy,eta,phi  ) ;
                                }
                            }

                            // third loop, cluster shapes to ensure the correct counts 
                            for(const auto& iterator :energy_for_cluster_shapes)
                            {
                                double energy = iterator.second / norm;
                                unsigned pid = iterator.first;
                                addToClusterShapes(cluster_container, pid, 0,energy,eta,phi,z  ) ;// only one for trigger cell 
                            }
                    } //end of for-loop
                }

                //3. Push the clusters in the cluster_product_
                for (auto&  p : cluster_container) 
                {
                    //std::unordered_map<uint64_t,std::pair<int,l1t::HGCalCluster> >
                    cluster_product_->push_back(p.second.first,p.second.second); // bx,cluster
                }

            } // end run


    }; // end class

}// namespace


// define plugins, template needs to be spelled out here, in order to allow the compiler to compile, and the factory to be populated
//
typedef HGCalTriggerBackend::HGCalTriggerSimCluster<HGCalTriggerCellBestChoiceCodec,HGCalTriggerCellBestChoiceDataPayload> HGCalTriggerSimClusterBestChoice;
typedef HGCalTriggerBackend::HGCalTriggerSimCluster<HGCalTriggerCellThresholdCodec,HGCalTriggerCellThresholdDataPayload> HGCalTriggerSimClusterThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimClusterBestChoice,"HGCalTriggerSimClusterBestChoice");
DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, HGCalTriggerSimClusterThreshold,"HGCalTriggerSimClusterThreshold");

