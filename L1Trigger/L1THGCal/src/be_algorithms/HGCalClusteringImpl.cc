#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"


//class constructor
HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet& conf){    
    seed_CUT_ = conf.getParameter<double>("seeding_threshold");
    tc_CUT_ = conf.getParameter<double>("clustering_threshold");
}
       

void  HGCalClusteringImpl::clusterizeBase(const l1t::HGCalTriggerCellBxCollection& trgcell_product_, l1t::HGCalClusterBxCollection& cluster_product_){
    
        double_t protoClEta = 0.;
        double_t protoClPhi = 0.;           
        double_t C2d_pt  = 0.;
        double_t C2d_eta = 0.;
        double_t C2d_phi = 0.;
        uint32_t C2d_hwPtEm = 0;
        uint32_t C2d_hwPtHad = 0;
                
        for(l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcell_product_.begin(); tc != trgcell_product_.end(); ++tc)
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
            }
        }
        l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), C2d_hwPtEm + C2d_hwPtHad, 0, 0);
        cluster.setHwPtEm(C2d_hwPtEm);
        cluster.setHwPtHad(C2d_hwPtHad);

        if((cluster.hwPtEm()+cluster.hwPtHad())>tc_CUT_){
            C2d_eta = protoClEta/C2d_pt;
            C2d_phi = protoClPhi/C2d_pt;                
            math::PtEtaPhiMLorentzVector calibP4(C2d_pt, C2d_eta, C2d_phi, 0 );
            cluster.setP4(calibP4);
            cluster_product_.push_back(0,cluster);
        }
    
} 
