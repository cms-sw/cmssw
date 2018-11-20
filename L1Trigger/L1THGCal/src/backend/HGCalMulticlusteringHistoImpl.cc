#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringHistoImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HGCalMulticlusteringHistoImpl::HGCalMulticlusteringHistoImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    dr_byLayer_coefficientA_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientA") ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientA") : std::vector<double>()),
    dr_byLayer_coefficientB_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientB") ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientB") : std::vector<double>()),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    multiclusterAlgoType_(conf.getParameter<string>("type_multicluster")),    
    cluster_association_input_(conf.getParameter<string>("cluster_association")),
    nBinsRHisto_(conf.getParameter<unsigned>("nBins_R_histo_multicluster")),
    nBinsPhiHisto_(conf.getParameter<unsigned>("nBins_Phi_histo_multicluster")),
    binsSumsHisto_(conf.getParameter< std::vector<unsigned> >("binSumsHisto")),
    histoThreshold_(conf.getParameter<double>("threshold_histo_multicluster")),
    neighbour_weights_(conf.getParameter< std::vector<double> >("neighbour_weights"))
{    
  
    if(multiclusterAlgoType_=="HistoMaxC3d"){
      multiclusteringAlgoType_ = HistoMaxC3d;
    }else if(multiclusterAlgoType_=="HistoSecondaryMaxC3d"){
      multiclusteringAlgoType_ = HistoSecondaryMaxC3d;
    }else if(multiclusterAlgoType_=="HistoThresholdC3d"){
      multiclusteringAlgoType_ = HistoThresholdC3d;
    }else if(multiclusterAlgoType_=="HistoInterpolatedMaxC3d"){
      multiclusteringAlgoType_ = HistoInterpolatedMaxC3d;
    }else {
      throw cms::Exception("HGCTriggerParameterError")
        << "Unknown Multiclustering type '" << multiclusterAlgoType_;
    } 
    
    if(cluster_association_input_=="NearestNeighbour"){
      cluster_association_strategy_ = NearestNeighbour;
    }else if(cluster_association_input_=="EnergySplit"){
      cluster_association_strategy_ = EnergySplit;
    }else {
      throw cms::Exception("HGCTriggerParameterError")
        << "Unknown cluster association strategy'" << cluster_association_strategy_;
    } 

    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR: " << dr_
    <<"\nMulticluster minimum transverse-momentum: " << ptC3dThreshold_
    <<"\nMulticluster number of R-bins for the histo algorithm: " << nBinsRHisto_
    <<"\nMulticluster number of Phi-bins for the histo algorithm: " << nBinsPhiHisto_
    <<"\nMulticluster MIPT threshold for histo threshold algorithm: " << histoThreshold_
    <<"\nMulticluster type of multiclustering algortihm: " << multiclusterAlgoType_;

    id_ = std::unique_ptr<HGCalTriggerClusterIdentificationBase>{ HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") };
    id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));
    if(multiclusterAlgoType_.find("Histo")!=std::string::npos && nBinsRHisto_!=binsSumsHisto_.size()){
      throw cms::Exception("Inconsistent bin size") <<  "Inconsistent nBins_R_histo_multicluster ( " << nBinsRHisto_ << " ) and binSumsHisto ( " << binsSumsHisto_.size() << " ) size in HGCalMulticlustering\n";
    }

    if(neighbour_weights_.size()!=neighbour_weights_size_) {
      throw cms::Exception("Inconsistent vector size" ) << "Inconsistent size of neighbour weights vector in HGCalMulticlustering ( " << neighbour_weights_.size() << " ). Should be " << neighbour_weights_size_ << "\n" ;
    }

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

        histoClusters[{{triggerTools_.zside(clu->detId()), bin_R, bin_phi}}]+=clu->mipPt();

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




std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeMaxSeeds( const Histogram & histoClusters ){

    std::vector<std::pair<GlobalPoint, double > > seedPositionsEnergy;

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;
                if (!isMax) continue;

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
                    seedPositionsEnergy.emplace_back( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed);

                }
            }

        }

    }

    return seedPositionsEnergy;

}


std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeInterpolatedMaxSeeds( const Histogram & histoClusters ){
  

  std::vector<std::pair<GlobalPoint, double > > seedPositionsEnergy;
  
    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){
              
                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;
                
                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;
                
                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;
                
                float MIPT_pred = neighbour_weights_.at(0) * MIPT_NW + neighbour_weights_.at(1) * MIPT_N + neighbour_weights_.at(2) * MIPT_NE
                  + neighbour_weights_.at(3) * MIPT_W + neighbour_weights_.at(5) * MIPT_E + neighbour_weights_.at(6) * MIPT_SW
                  + neighbour_weights_.at(7) * MIPT_S + neighbour_weights_.at(8) * MIPT_SE;
                
                bool isMax = MIPT_seed>=(MIPT_pred+histoThreshold_);
                
                if(isMax){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositionsEnergy.emplace_back(  GlobalPoint(x_seed,y_seed,z_side), MIPT_seed);
                }
                
            }
            
        }
        
    }
    
    return seedPositionsEnergy;
    
}


std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeThresholdSeeds( const Histogram & histoClusters ){


    std::vector<std::pair<GlobalPoint, double > > seedPositionsEnergy;

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
                    seedPositionsEnergy.emplace_back(  GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) ;
                }

            }

        }

    }

    return seedPositionsEnergy;

}



std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeSecondaryMaxSeeds( const Histogram & histoClusters ){

    std::vector<std::pair<GlobalPoint, double > > seedPositionsEnergy;
  
    std::map<std::tuple<int,int,int>, bool> primarySeedPositions;
    std::map<std::tuple<int,int,int>, bool> secondarySeedPositions;
    std::map<std::tuple<int,int,int>, bool> vetoPositions;

    //Search for primary seeds
    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;

                if (!isMax) continue;

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

                    seedPositionsEnergy.emplace_back( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed );
                    primarySeedPositions[std::make_tuple(bin_R,bin_phi,z_side)] =  true;

                    vetoPositions[std::make_tuple(bin_R,binLeft,z_side)] = true;
                    vetoPositions[std::make_tuple(bin_R,binRight,z_side)] = true;
                    if ( bin_R>0 ) {
                        vetoPositions[std::make_tuple(bin_R-1,bin_phi,z_side)] = true;
                        vetoPositions[std::make_tuple(bin_R-1,binRight,z_side)] = true;
                        vetoPositions[std::make_tuple(bin_R-1,binLeft,z_side)] = true;
                    }
                    if ( bin_R<(int(nBinsRHisto_)-1) ) {
                        vetoPositions[std::make_tuple(bin_R+1,bin_phi,z_side)] = true;
                        vetoPositions[std::make_tuple(bin_R+1,binRight,z_side)] = true;
                        vetoPositions[std::make_tuple(bin_R+1,binLeft,z_side)] = true;
                    }

                }

            }

        }

    }


    //Search for secondary seeds

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                //Cannot be a secondary seed if already a primary seed, or adjacent to primary seed
                if ( primarySeedPositions[std::make_tuple(bin_R,bin_phi,z_side)] || vetoPositions[std::make_tuple(bin_R,bin_phi,z_side)] ) continue;

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;
                
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


                if (  !vetoPositions[std::make_tuple(bin_R+1,bin_phi,z_side)]  ) isMax &= MIPT_seed>=MIPT_S;
                if (  !vetoPositions[std::make_tuple(bin_R-1,bin_phi,z_side)]  ) isMax &= MIPT_seed>MIPT_N;
                if (  !vetoPositions[std::make_tuple(bin_R,binRight,z_side)]  ) isMax &= MIPT_seed>=MIPT_E;
                if (  !vetoPositions[std::make_tuple(bin_R+1,binRight,z_side)]  ) isMax &= MIPT_seed>=MIPT_SE;
                if (  !vetoPositions[std::make_tuple(bin_R-1,binRight,z_side)]  ) isMax &= MIPT_seed>=MIPT_NE;
                if (  !vetoPositions[std::make_tuple(bin_R,binLeft,z_side)]  ) isMax &= MIPT_seed>MIPT_W;
                if (  !vetoPositions[std::make_tuple(bin_R+1,binLeft,z_side)]  ) isMax &= MIPT_seed>MIPT_SW;
                if (  !vetoPositions[std::make_tuple(bin_R-1,binLeft,z_side)]  ) isMax &= MIPT_seed>MIPT_NW;

                if(isMax){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositionsEnergy.emplace_back( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed );
                    secondarySeedPositions[std::make_tuple(bin_R,bin_phi,z_side)] =  true;
                }

            }

        }

    }

    return seedPositionsEnergy;

}



std::vector<l1t::HGCalMulticluster> HGCalMulticlusteringHistoImpl::clusterSeedMulticluster(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
                                                                                           const std::vector<std::pair<GlobalPoint, double> > & seeds){

    std::map<int,l1t::HGCalMulticluster> mapSeedMulticluster;
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    for(auto & clu : clustersPtrs){

        int z_side = triggerTools_.zside(clu->detId());


        double radiusCoefficientA = dr_byLayer_coefficientA_.empty() ? dr_ : dr_byLayer_coefficientA_.at(triggerTools_.layerWithOffset(clu->detId())); // use at() to get the assert, for the moment
        double radiusCoefficientB = dr_byLayer_coefficientB_.empty() ? 0 : dr_byLayer_coefficientB_.at(triggerTools_.layerWithOffset(clu->detId())); // use at() to get the assert, for the moment

        double minDist = radiusCoefficientA + radiusCoefficientB*(kMidRadius_ - std::abs(clu->eta()) ) ;

        std::vector<pair<int,double> > targetSeedsEnergy;
    
        for( unsigned int iseed=0; iseed<seeds.size(); iseed++ ){

            GlobalPoint seedPosition = seeds[iseed].first;          
            double seedEnergy = seeds[iseed].second;        

            if( z_side*seedPosition.z()<0) continue;
            double d = this->dR(*clu, seeds[iseed].first);

            if ( d < minDist ){
                if ( cluster_association_strategy_ == EnergySplit ){
                    targetSeedsEnergy.emplace_back( iseed, seedEnergy );
                }
                if ( cluster_association_strategy_ == NearestNeighbour ){

                    minDist = d;

                    if ( targetSeedsEnergy.empty() ) {
                        targetSeedsEnergy.emplace_back( iseed, seedEnergy );
                    }
                    else {
                        targetSeedsEnergy.at(0).first = iseed ;
                        targetSeedsEnergy.at(0).second = seedEnergy;
                    }
                }
                
            }
            
        }
        
        if(targetSeedsEnergy.empty()) continue;
        //Loop over target seeds and divide up the clusters energy
        double totalTargetSeedEnergy = 0;
        for (auto energy: targetSeedsEnergy){
            totalTargetSeedEnergy+=energy.second;
        }
    
        for (auto energy: targetSeedsEnergy){

            double seedWeight = 1;
            if ( cluster_association_strategy_ == EnergySplit) seedWeight = energy.second/totalTargetSeedEnergy;
            if( mapSeedMulticluster[energy.first].size()==0) {
                mapSeedMulticluster[energy.first] = l1t::HGCalMulticluster(clu, seedWeight) ;
            }
            mapSeedMulticluster[energy.first].addConstituent(clu, true, seedWeight);   
          
        }
        
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
    std::vector<std::pair<GlobalPoint, double> > seedPositionsEnergy;
    if (multiclusteringAlgoType_ == HistoMaxC3d) seedPositionsEnergy = computeMaxSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoThresholdC3d) seedPositionsEnergy = computeThresholdSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoInterpolatedMaxC3d) seedPositionsEnergy = computeInterpolatedMaxSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoSecondaryMaxC3d) seedPositionsEnergy = computeSecondaryMaxSeeds(smoothRPhiHistoCluster);
    /* clusterize clusters around seeds */
    std::vector<l1t::HGCalMulticluster> multiclustersTmp = clusterSeedMulticluster(clustersPtrs,seedPositionsEnergy);    
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
        double sumPt=multicluster.sumPt();

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
            // fill H/E
            multicluster.saveHOverE();            

            multiclusters_out.push_back( 0, multicluster);
        }
    }
}
