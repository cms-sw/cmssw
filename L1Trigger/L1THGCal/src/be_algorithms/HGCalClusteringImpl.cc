#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"


HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet & beCodecConfig){    
    
    seedThr_ = beCodecConfig.getParameter<double>("seeding_threshold");
    tcThr_ = beCodecConfig.getParameter<double>("clustering_threshold");
    dr_ = beCodecConfig.getParameter<double>("dR_cluster");

}


void HGCalClusteringImpl::clusterise( const l1t::HGCalTriggerCellBxCollection & trgcells_, 
                                      l1t::HGCalClusterBxCollection & clusters_
    ){
    
    double_t protoClEta = 0.;
    double_t protoClPhi = 0.;           
    double_t C2d_pt  = 0.;
    double_t C2d_eta = 0.;
    double_t C2d_phi = 0.;
    uint32_t C2d_hwPtEm = 0;
    uint32_t C2d_hwPtHad = 0;
        
    std::cout << "CLUSTERING PARAMETERS: "<< std::endl;
    std::cout << "------ Clustering thresholds for trigger cells to be included in C2d: " << tcThr_ << std::endl;
    std::cout << "------ Seeding thresholds to start the clusterization procedure: " << seedThr_ << std::endl; 

    int layer=0;
    bool seeds[trgcells_.size()];

    int itc=0;
    for(l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells_.begin(); tc != trgcells_.end(); ++tc,++itc)
        seeds[itc] = (tc->hwPt() > seedThr_) ? true : false;

    itc=0;
    for(l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells_.begin(); tc != trgcells_.end(); ++tc,++itc){

        if( seeds[itc] ){}
        
        int iclu=0;
        vector<int> tcPertinentClusters; 
        for(l1t::HGCalClusterBxCollection::const_iterator clu = clusters_.begin(); clu != clusters_.end(); ++clu,++iclu){
            if( clu->isPertinent(*tc, dr_) )
                tcPertinentClusters.push_back(iclu);
        }

        if( tcPertinentClusters.size() == 0 ){
            l1t::HGCalCluster obj( *tc );
            clusters_.push_back( 0, obj );
        }
        else{
            uint minDist = 1;
            uint targetClu = 0; 
            for( std::vector<int>::const_iterator iclu = tcPertinentClusters.begin(); iclu != tcPertinentClusters.end(); ++iclu ){
                //double d = clusters_.at(0, *iclu)->dist(tc);
//vito                //if( d < minDist ){
//vito                    //  minDist = d;
//vito                    //targetClu = *iclu;
//vito                //}
            } 
//vito
//vito            clusters_.at(targetClu)->addTC(tc);
//vito
        }
       
    }
        
    //       if(tc->hwPt()>0)
    //       {
    //                       
 //           HGCalDetId trgdetid(tc->detId());                
 //           int trgCellLayer = trgdetid.layer();               
 //                       
 //           if(trgCellLayer<28){
 //               C2d_hwPtEm+=tc->hwPt();
 //           }else if(trgCellLayer>=28){
 //               C2d_hwPtHad+=tc->hwPt();
 //           }
 //                              
 //           C2d_pt += tc->pt();                        
 //           protoClEta += tc->pt()*tc->eta();
 //           protoClPhi += tc->pt()*tc->phi();
 //           //CODE THE REAL C2D-ALGORITHM HERE: using trg-cells + neighbours info
 //               
 //       }
 //
 //       l1t::HGCalCluster cluster( reco::LeafCandidate::LorentzVector(), C2d_hwPtEm + C2d_hwPtHad, 0, 0);
 //    
 //       cluster.setHwPtEm(C2d_hwPtEm);
 //       cluster.setHwPtHad(C2d_hwPtHad);
 //
 //       if((cluster.hwPtEm()+cluster.hwPtHad())>tcThr_){
 //           C2d_eta = protoClEta/C2d_pt;
 //           C2d_phi = protoClPhi/C2d_pt;                
 //           math::PtEtaPhiMLorentzVector calibP4(C2d_pt, C2d_eta, C2d_phi, 0 );
 //           cluster.setP4(calibP4);
 //           clusters_.push_back(0,cluster);
 //           //std::cout << "Energy of the uncalibrated cluster " << C2d_hwPtEm + C2d_hwPtHad << "  with EM-pt() = " << cluster.hwPtEm()<< " had-pt = "<<cluster.hwPtHad() <<"   id-module " << cluster.module() << "  layer " << cluster.layer() << std::endl ; //use pt and not pt()
 //           //std::cout << "    ----> 4P of C2d (pt,eta,phi,M) = " << cluster.p4().Pt()<<", " << cluster.p4().Eta() << ", " << cluster.p4().Phi() << ", " << cluster.p4().M() << std::endl;
 //       }
 //   
 //   
 //       // return clusters_;
 //   }

}

