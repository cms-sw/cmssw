#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"


//class constructor
HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet& beCodecConfig){    
    seed_CUT_ = beCodecConfig.getParameter<double>("seeding_threshold");
    tc_CUT_ = beCodecConfig.getParameter<double>("clustering_threshold");
}
       

void  HGCalClusteringImpl::clusterizeBase(std::unique_ptr<l1t::HGCalTriggerCellBxCollection>& trgcell_product_, std::unique_ptr<l1t::HGCalClusterBxCollection> & cluster_product_){
    
        double_t protoClEta = 0.;
        double_t protoClPhi = 0.;           
        double_t C2d_pt  = 0.;
        double_t C2d_eta = 0.;
        double_t C2d_phi = 0.;
        uint32_t C2d_hwPtEm = 0;
        uint32_t C2d_hwPtHad = 0;
                
//        assert trgcell_product_->size() > 0 //&& put logWarning

        for(l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcell_product_->begin(); tc != trgcell_product_->end(); ++tc)
        {
            if(tc->hwPt()>0)
            {
                        
                HGCalDetId trgdetid(tc->detId());                
                int trgCellLayer = trgdetid.layer();               
                        
                if(trgCellLayer<28){
                    C2d_hwPtEm+=tc->hwPt();
                }else if(trgCellLayer>=28){
                    C2d_hwPtHad+=tc->hwPt();
                }
                               
                C2d_pt += tc->pt();                        
                protoClEta += tc->pt()*tc->eta();
                protoClPhi += tc->pt()*tc->phi();
                //CODE THE REAL C2D-ALGORITHM HERE: using trg-cells + neighbours info
            }
        }
        l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), C2d_hwPtEm + C2d_hwPtHad, 0, 0);
        //cluster.setModule(trgdetid.wafer());
        //cluster.setLayer(trgdetid.layer());
        //cluster.setSubDet(trgdetid.subdetId());
        cluster.setHwPtEm(C2d_hwPtEm);
        cluster.setHwPtHad(C2d_hwPtHad);

        if((cluster.hwPtEm()+cluster.hwPtHad())>tc_CUT_){
            C2d_eta = protoClEta/C2d_pt;
            C2d_phi = protoClPhi/C2d_pt;                
            math::PtEtaPhiMLorentzVector calibP4(C2d_pt, C2d_eta, C2d_phi, 0 );
            cluster.setP4(calibP4);
            cluster_product_->push_back(0,cluster);
            //std::cout << "Energy of the uncalibrated cluster " << C2d_hwPtEm + C2d_hwPtHad << "  with EM-pt() = " << cluster.hwPtEm()<< " had-pt = "<<cluster.hwPtHad() <<"   id-module " << cluster.module() << "  layer " << cluster.layer() << std::endl ; //use pt and not pt()
            //std::cout << "    ----> 4P of C2d (pt,eta,phi,M) = " << cluster.p4().Pt()<<", " << cluster.p4().Eta() << ", " << cluster.p4().Phi() << ", " << cluster.p4().M() << std::endl;
        }
    
    
    // return cluster_product_;
} 
