#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster3D.h"

using namespace HGCalTriggerBackend;

template<typename FECODEC, typename DATA>
class C2dClusterAlgo : public Algorithm<FECODEC> 
{
    public:
        using Algorithm<FECODEC>::name;

    protected:
        using Algorithm<FECODEC>::codec_;

    public:
        C2dClusterAlgo(const edm::ParameterSet& conf):
        Algorithm<FECODEC>(conf),
        cluster_product_( new l1t::HGCalClusterBxCollection ),
        cluster3D_product_( new l1t::HGCalCluster3DBxCollection ),
        HGCalEESensitive_(conf.getParameter<std::string>("HGCalEESensitive_tag")),
        HGCalHESiliconSensitive_(conf.getParameter<std::string>("HGCalHESiliconSensitive_tag")),
        calibration_(conf.getParameterSet("calib_parameters")),
        seed_CUT_(conf.getParameter<double>("seeding_threshold")), 
        tc_CUT_(conf.getParameter<double>("clustering_threshold")),
        dR_forC3d_(conf.getParameter<double>("dR_searchNeighbour")){}

        typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
            prod.produces<l1t::HGCalCluster3DBxCollection>("cluster3D");

        }
    
        virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) override final
        {
            es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
            es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);
            std::cout << "CLUSTERING PARAMETERS: "<< std::endl;
            std::cout << "------ Clustering thresholds for trigger cells to be included in C2d: " << tc_CUT_ << std::endl;
            std::cout << "------ Seeding thresholds to start the clusterization procedure: " << seed_CUT_ << std::endl; 
            std::cout << "------ Max-Distance in the normalized plane to search for next-layer C2d to merge into the C3d: " << dR_forC3d_ << std::endl;  
            
            for( const auto& digi : coll ) 
            {

                DATA data;
                data.reset();
                const HGCalDetId& module_id = digi.getDetId<HGCalDetId>();
                digi.decode(codec_, data);
                
                double_t moduleEta = 0.;
                double_t modulePhi = 0.;           
                double_t C2d_pt  = 0.;
                double_t C2d_eta = 0.;
                double_t C2d_phi = 0.;
                uint32_t C2d_hwPtEm = 0;
                uint32_t C2d_hwPtHad = 0;
                for(const auto& triggercell : data.payload)
                {
                    if(triggercell.hwPt()>0)
                    {
                        
                        HGCalDetId detid(triggercell.detId());
                        int subdet = detid.subdetId();
                        int cellThickness = 0;
                        
                        if( subdet == HGCEE ){ 
                            cellThickness = (hgceeTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                        }else if( subdet == HGCHEF ){
                            cellThickness = (hgchefTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                        }else if( subdet == HGCHEB ){
                            edm::LogWarning("DataNotFound") << "ATTENTION: the BH trgCells are not yet implemented !! ";
                        }
              
                        if(module_id.layer()<28){
                            C2d_hwPtEm+=triggercell.hwPt();
                        }else if(module_id.layer()>=28){
                            C2d_hwPtHad+=triggercell.hwPt();
                        }
              
                        l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                        calibration_.calibrate(calibratedtriggercell, cellThickness);     
                        C2d_pt += calibratedtriggercell.pt();                        
                        moduleEta += calibratedtriggercell.pt()*calibratedtriggercell.eta();
                        modulePhi += calibratedtriggercell.pt()*calibratedtriggercell.phi();
                        //CODE THE REAL C2D-ALGORITHM HERE: using trg-cells + neighbours info
                    }
                }
                l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), C2d_hwPtEm + C2d_hwPtHad, 0, 0);
                cluster.setModule(module_id.wafer());
                cluster.setLayer(module_id.layer());
                cluster.setSubDet(module_id.subdetId());
                cluster.setHwPtEm(C2d_hwPtEm);
                cluster.setHwPtHad(C2d_hwPtHad);

                if((cluster.hwPtEm()+cluster.hwPtHad())>tc_CUT_+8){
                    C2d_eta = moduleEta/C2d_pt;
                    C2d_phi = modulePhi/C2d_pt;                
                    math::PtEtaPhiMLorentzVector calibP4(C2d_pt, C2d_eta, C2d_phi, 0 );
                    cluster.setP4(calibP4);
                    cluster_product_->push_back(0,cluster);
                    std::cout << "Energy of the uncalibrated cluster " << C2d_hwPtEm + C2d_hwPtHad << "  with EM-pt() = " << cluster.hwPtEm()<< " had-pt = "<<cluster.hwPtHad() <<"   id-module " << cluster.module() << "  layer " << cluster.layer() << std::endl ; //use pt and not pt()
                    std::cout << "    ----> 4P of C2d (pt,eta,phi,M) = " << cluster.p4().Pt()<<", " << cluster.p4().Eta() << ", " << cluster.p4().Phi() << ", " << cluster.p4().M() << std::endl;
                }
            }
            //CODE THE REAL C3D-ALGORITHM HERE: using previous built C2D  + fill all information transmittable to the CORRELATOR                
            std::vector<size_t> isMerged;
            l1t::HGCalCluster3D cluster3D( reco::LeafCandidate::LorentzVector(), 0, 0, 0);
            double_t tmpEta = 0.;
            double_t tmpPhi = 0.;           
            double_t C3d_pt  = 0.;
            double_t C3d_eta = 0.;
            double_t C3d_phi = 0.;
            uint32_t C3d_hwPtEm = 0;
            uint32_t C3d_hwPtHad = 0;
            
            for(l1t::HGCalClusterBxCollection::const_iterator c2d = cluster_product_->begin(); c2d != cluster_product_->end(); ++c2d){
                std::cout << "looping on the c2d directly from the collection : "<< c2d->p4().pt() << " eta: " <<  c2d->p4().Eta() << " --> layer " << c2d->layer() << std::endl;                
                
                bool skip=false;
                size_t idx=0;
                for(l1t::HGCalClusterBxCollection::const_iterator c2d_aux = cluster_product_->begin(); c2d_aux != cluster_product_->end(); ++c2d_aux, ++idx){
                    for(size_t i(0); i<isMerged.size(); i++){
                        if(idx==isMerged.at(i)){
                            skip=true;
                            continue;
                        }
                    }

                    if(skip) continue;
                    //std::cout << "looping on the c2d directly from the collection : "<< c2d->p4().pt() << "  --> layer " << c2d->layer() << std::endl;
                    if( deltaR( c2d->p4().Eta(), c2d_aux->p4().Eta(), c2d->p4().Phi(), c2d_aux->p4().Phi() ) < dR_forC3d_  ){
                        isMerged.push_back(idx);
                        tmpEta+=c2d_aux->p4().Eta() * c2d_aux->p4().Pt();
                        tmpPhi+=c2d_aux->p4().Phi() * c2d_aux->p4().Pt();
                        C3d_pt+=c2d_aux->p4().Pt();
                        C3d_hwPtEm+=c2d_aux->hwPtEm();
                        C3d_hwPtHad+=c2d_aux->hwPtHad();
                        
                    }
                }
                
                if((cluster3D.hwPtEm()+cluster3D.hwPtHad()) > 0){
                    cluster3D.setHwPtEm(C3d_hwPtEm);
                    cluster3D.setHwPtHad(C3d_hwPtHad);
                    C3d_eta=tmpEta/C3d_pt;
                    C3d_phi=tmpPhi/C3d_pt;                
                    math::PtEtaPhiMLorentzVector calib3dP4(C3d_pt, C3d_eta, C3d_phi, 0 );
                    cluster3D.setP4(calib3dP4);                    
                    cluster3D_product_->push_back(0,cluster3D);
                }
            }
        }
    
        virtual void putInEvent(edm::Event& evt) override final 
        {
            evt.put(std::move(cluster_product_),name());
            evt.put(std::move(cluster3D_product_),"cluster3D");
        }

        virtual void reset() override final 
        {
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );            
            cluster3D_product_.reset( new l1t::HGCalCluster3DBxCollection );            
        }

    private:

        std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;
        std::unique_ptr<l1t::HGCalCluster3DBxCollection> cluster3D_product_;
        std::string HGCalEESensitive_;
        std::string HGCalHESiliconSensitive_;

        edm::ESHandle<HGCalTopology> hgceeTopoHandle_;
        edm::ESHandle<HGCalTopology> hgchefTopoHandle_;
        HGCalTriggerCellCalibration calibration_;    
        double seed_CUT_;
        double tc_CUT_;
        double dR_forC3d_;

        double deltaPhi( double phi1, double phi2) {
        
           double dPhi(phi1-phi2);
           double pi(acos(-1.0));
           if     (dPhi<=-pi) dPhi+=2.0*pi;
           else if(dPhi> pi) dPhi-=2.0*pi;
        
           return dPhi;
        }
    

        double deltaEta(double eta1, double eta2){
           double dEta = (eta1-eta2);
           return dEta;
        }

        double deltaR(double eta1, double eta2, double phi1, double phi2) {
           double dEta = deltaEta(eta1, eta2);
           double dPhi = deltaPhi(phi1, phi2);
           return sqrt(dEta*dEta+dPhi*dPhi);
        }

};

typedef C2dClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> C2dClusterAlgoBestChoice;
typedef C2dClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> C2dClusterAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        C2dClusterAlgoBestChoice,
        "C2dClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        C2dClusterAlgoThreshold,
        "C2dClusterAlgoThreshold");
