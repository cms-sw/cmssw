#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"

//class constructor
HGCalMulticlusteringImpl::HGCalMulticlusteringImpl(const edm::ParameterSet& beCodecConfig){    
    dR_forC3d_ = beCodecConfig.getParameter<double>("dR_searchNeighbour");
}
       
void  HGCalMulticlusteringImpl::clusterizeMultiple(std::unique_ptr<l1t::HGCalClusterBxCollection> & cluster_product_, std::unique_ptr<l1t::HGCalMulticlusterBxCollection> & multicluster_product_){
    

    if(cluster_product_->size()>0){
        std::vector<size_t> isMerged;

        size_t seedx=0;
 
        for(l1t::HGCalClusterBxCollection::const_iterator cl = cluster_product_->begin(); cl != cluster_product_->end(); ++cl, ++seedx){
            edm::PtrVector<l1t::HGCalCluster> ClusterCollection;
        
            l1t::HGCalMulticluster multicluster( reco::LeafCandidate::LorentzVector(), 0, 0, 0, ClusterCollection);
            double_t tmpEta = 0.;
            double_t tmpPhi = 0.;           
            double_t C3d_pt  = 0.;
            double_t C3d_eta = 0.;
            double_t C3d_phi = 0.;
            uint32_t C3d_hwPtEm = 0;
            uint32_t C3d_hwPtHad = 0;
            uint32_t totLayer = 0;

            bool skip=false;
            size_t idx=0;
            for(l1t::HGCalClusterBxCollection::const_iterator cl_aux = cluster_product_->begin(); cl_aux != cluster_product_->end(); ++cl_aux, ++idx){
                for(size_t i(0); i<isMerged.size(); i++){
                    if(idx==isMerged.at(i)){
                        skip=true;
                        continue;
                    }
                }
                double dR = deltaR(cl->p4(), cl_aux->p4());
            
                if(skip){
                    skip=false;
                    continue;
                }
                if( dR < dR_forC3d_*10 ){
                    isMerged.push_back(idx);
                    tmpEta+=cl_aux->p4().Eta() * cl_aux->p4().Pt();
                    tmpPhi+=cl_aux->p4().Phi() * cl_aux->p4().Pt();
                    C3d_pt+=cl_aux->p4().Pt();
                    C3d_hwPtEm+=cl_aux->hwPtEm();
                    C3d_hwPtHad+=cl_aux->hwPtHad();
                    totLayer++;
                }
            }
        
            if( totLayer > 2){
                edm::PtrVector<l1t::HGCalCluster> ClusterCollection; //push_back()
                multicluster.setNtotLayer(totLayer);
                multicluster.setHwPtEm(C3d_hwPtEm);
                multicluster.setHwPtHad(C3d_hwPtHad);
                C3d_eta=tmpEta/C3d_pt;
                C3d_phi=tmpPhi/C3d_pt;                
                math::PtEtaPhiMLorentzVector calib3dP4(C3d_pt, C3d_eta, C3d_phi, 0 );
                multicluster.setP4(calib3dP4);                    
                multicluster_product_->push_back(0,multicluster);
            }                    
        }
    }
} 
