#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringHistoImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HGCalMulticlusteringHistoImpl::HGCalMulticlusteringHistoImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    multiclusterAlgoType_(conf.getParameter<string>("type_multicluster")),    
    nBinsRHisto_(conf.getParameter<unsigned>("nBins_R_histo_multicluster")),
    nBinsPhiHisto_(conf.getParameter<unsigned>("nBins_Phi_histo_multicluster")),
    binsSumsHisto_(conf.getParameter< std::vector<unsigned> >("binSumsHisto")),
    histoThreshold_(conf.getParameter<double>("threshold_histo_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster number of R-bins for the histo algorithm: " << nBinsRHisto_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster number of Phi-bins for the histo algorithm: " << nBinsPhiHisto_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster MIPT threshold for histo threshold algorithm: " << histoThreshold_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster type of multiclustering algortihm: " << multiclusterAlgoType_;
    id_.reset( HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") );
    id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));
    if(multiclusterAlgoType_.find("Histo")!=std::string::npos && nBinsRHisto_!=binsSumsHisto_.size()) throw cms::Exception("Inconsistent nBins_R_histo_multicluster and binSumsHisto size in HGCalMulticlustering");
}





float HGCalMulticlusteringHistoImpl::dR( const l1t::HGCalCluster & clu,
					 const GlobalPoint & seed) const
{

    Basic3DVector<float> seed_3dv( seed );
    GlobalPoint seed_proj( seed_3dv / seed.z() );
    return (seed_proj - clu.centreProj() ).mag();

}





HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillHistoClusters( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs ){


    Histogram histoClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                histoClusters[{{z_side, bin_R, bin_phi}}] = 0;

            }

        }

    }


    for(auto & clu : clustersPtrs){

        float ROverZ = sqrt( pow(clu->centreProj().x(),2) + pow(clu->centreProj().y(),2) );
        int bin_R = int( (ROverZ-kROverZMin_) * nBinsRHisto_ / (kROverZMax_-kROverZMin_) );
        int bin_phi = int( (reco::reduceRange(clu->phi())+M_PI) * nBinsPhiHisto_ / (2*M_PI) );

        histoClusters[{{clu->zside(), bin_R, bin_phi}}]+=clu->mipPt();

    }

    return histoClusters;

}




HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillSmoothPhiHistoClusters( const Histogram & histoClusters,
								     const vector<unsigned> & binSums ){

    Histogram histoSumPhiClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            int nBinsSide = (binSums[bin_R]-1)/2;
            float R1 = kROverZMin_ + bin_R*(kROverZMax_-kROverZMin_);
            float R2 = R1 + (kROverZMax_-kROverZMin_);
            double area = 0.5 * (pow(R2,2)-pow(R1,2)) * (1+0.5*(1-pow(0.5,nBinsSide))); // Takes into account different area of bins in different R-rings + sum of quadratic weights used

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float content = histoClusters.at({{z_side,bin_R,bin_phi}});

                for(int bin_phi2=1; bin_phi2<=nBinsSide; bin_phi2++ ){

                    int binToSumLeft = bin_phi - bin_phi2;
                    if( binToSumLeft<0 ) binToSumLeft += nBinsPhiHisto_;
                    int binToSumRight = bin_phi + bin_phi2;
                    if( binToSumRight>=int(nBinsPhiHisto_) ) binToSumRight -= nBinsPhiHisto_;

                    content += histoClusters.at({{z_side,bin_R,binToSumLeft}}) / pow(2,bin_phi2); // quadratic kernel
                    content += histoClusters.at({{z_side,bin_R,binToSumRight}}) / pow(2,bin_phi2); // quadratic kernel

                }

                histoSumPhiClusters[{{z_side,bin_R,bin_phi}}] = content/area;

            }

        }

    }

    return histoSumPhiClusters;

}






HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillSmoothRPhiHistoClusters( const Histogram & histoClusters ){

    Histogram histoSumRPhiClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            float weight = (bin_R==0 || bin_R==int(nBinsRHisto_)-1) ? 1.5 : 2.; //Take into account edges with only one side up or down

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float content = histoClusters.at({{z_side,bin_R,bin_phi}});
                float contentDown = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0 ;
                float contentUp = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;

                histoSumRPhiClusters[{{z_side,bin_R,bin_phi}}] = (content + 0.5*contentDown + 0.5*contentUp)/weight;

            }

        }

    }

    return histoSumRPhiClusters;

}





std::vector<GlobalPoint> HGCalMulticlusteringHistoImpl::computeMaxSeeds( const Histogram & histoClusters ){

    std::vector<GlobalPoint> seedPositions;

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed>0;

                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;

                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;

                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ?histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;

                isMax &= MIPT_seed>=MIPT_S;
                isMax &= MIPT_seed>MIPT_N;
                isMax &= MIPT_seed>=MIPT_E;
                isMax &= MIPT_seed>=MIPT_SE;
                isMax &= MIPT_seed>=MIPT_NE;
                isMax &= MIPT_seed>MIPT_W;
                isMax &= MIPT_seed>MIPT_SW;
                isMax &= MIPT_seed>MIPT_NW;

                if(isMax){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositions.emplace_back(x_seed,y_seed,z_side);
                }

            }

        }

    }

    return seedPositions;

}





std::vector<GlobalPoint> HGCalMulticlusteringHistoImpl::computeThresholdSeeds( const Histogram & histoClusters ){

    std::vector<GlobalPoint> seedPositions;

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isSeed = MIPT_seed > histoThreshold_;

                if(isSeed){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositions.emplace_back(x_seed,y_seed,z_side);
                }

            }

        }

    }

    return seedPositions;

}



std::vector<l1t::HGCalMulticluster> HGCalMulticlusteringHistoImpl::clusterSeedMulticluster(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
											   const std::vector<GlobalPoint> & seeds){


    std::map<int,l1t::HGCalMulticluster> mapSeedMulticluster;
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    for(auto & clu : clustersPtrs){


        HGCalDetId cluDetId( clu->detId() );
        int z_side = cluDetId.zside();

        double minDist = dr_;
        int targetSeed = -1;

        for( unsigned int iseed=0; iseed<seeds.size(); iseed++ ){

            if( z_side*seeds[iseed].z()<0) continue;

            double d = this->dR(*clu, seeds[iseed]);

            if(d<minDist){
                minDist = d;
                targetSeed = iseed;
            }

        }

        if(targetSeed<0) continue;

        if(mapSeedMulticluster[targetSeed].size()==0) mapSeedMulticluster[targetSeed] = l1t::HGCalMulticluster(clu);
        else mapSeedMulticluster[targetSeed].addConstituent(clu);

    }

    for(auto mclu : mapSeedMulticluster) multiclustersTmp.emplace_back(mclu.second);

    return multiclustersTmp;

}




void HGCalMulticlusteringHistoImpl::clusterizeHisto( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
						     l1t::HGCalMulticlusterBxCollection & multiclusters,
						     const HGCalTriggerGeometryBase & triggerGeometry)
{

    /* put clusters into an r/z x phi histogram */
    Histogram histoCluster = fillHistoClusters(clustersPtrs); //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi, content = MIPTs summed along depth

    /* smoothen along the phi direction + normalize each bin to same area */
    Histogram smoothPhiHistoCluster = fillSmoothPhiHistoClusters(histoCluster,binsSumsHisto_);

    /* smoothen along the r/z direction */
    Histogram smoothRPhiHistoCluster = fillSmoothRPhiHistoClusters(histoCluster);

    /* seeds determined with local maximum criteria */
    std::vector<GlobalPoint> seedPositions;
    if(multiclusterAlgoType_ == "HistoMaxC3d") seedPositions = computeMaxSeeds(smoothRPhiHistoCluster);
    else if(multiclusterAlgoType_ == "HistoThresholdC3d") seedPositions = computeThresholdSeeds(smoothRPhiHistoCluster);

    /* clusterize clusters around seeds */
    std::vector<l1t::HGCalMulticluster> multiclustersTmp = clusterSeedMulticluster(clustersPtrs,seedPositions);

    /* making the collection of multiclusters */
    finalizeClusters(multiclustersTmp, multiclusters, triggerGeometry);

}







void
HGCalMulticlusteringHistoImpl::
finalizeClusters(std::vector<l1t::HGCalMulticluster>& multiclusters_in,
            l1t::HGCalMulticlusterBxCollection& multiclusters_out, 
            const HGCalTriggerGeometryBase& triggerGeometry) {
    for(auto& multicluster : multiclusters_in) {
        // compute the eta, phi from its barycenter
        // + pT as scalar sum of pT of constituents
        double sumPt=0.;
        const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clusters = multicluster.constituents();
        for(const auto& id_cluster : clusters) sumPt += id_cluster.second->pt();

        math::PtEtaPhiMLorentzVector multiclusterP4(  sumPt,
                multicluster.centre().eta(),
                multicluster.centre().phi(),
                0. );
        multicluster.setP4( multiclusterP4 );

        if( multicluster.pt() > ptC3dThreshold_ ){
            //compute shower shapes
            multicluster.showerLength(shape_.showerLength(multicluster));
            multicluster.coreShowerLength(shape_.coreShowerLength(multicluster, triggerGeometry));
            multicluster.firstLayer(shape_.firstLayer(multicluster));
            multicluster.maxLayer(shape_.maxLayer(multicluster));
            multicluster.sigmaEtaEtaTot(shape_.sigmaEtaEtaTot(multicluster));
            multicluster.sigmaEtaEtaMax(shape_.sigmaEtaEtaMax(multicluster));
            multicluster.sigmaPhiPhiTot(shape_.sigmaPhiPhiTot(multicluster));
            multicluster.sigmaPhiPhiMax(shape_.sigmaPhiPhiMax(multicluster));
            multicluster.sigmaZZ(shape_.sigmaZZ(multicluster));
            multicluster.sigmaRRTot(shape_.sigmaRRTot(multicluster));
            multicluster.sigmaRRMax(shape_.sigmaRRMax(multicluster));
            multicluster.sigmaRRMean(shape_.sigmaRRMean(multicluster));
            multicluster.eMax(shape_.eMax(multicluster));
            // fill quality flag
            multicluster.setHwQual(id_->decision(multicluster));

            multiclusters_out.push_back( 0, multicluster);
        }
    }
}
