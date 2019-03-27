#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HGCalHistoClusteringImpl::HGCalHistoClusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    dr_byLayer_coefficientA_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientA") ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientA") : std::vector<double>()),
    dr_byLayer_coefficientB_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientB") ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientB") : std::vector<double>()),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    cluster_association_input_(conf.getParameter<string>("cluster_association"))
{    
  
    if(cluster_association_input_=="NearestNeighbour"){
      cluster_association_strategy_ = NearestNeighbour;
    }else if(cluster_association_input_=="EnergySplit"){
      cluster_association_strategy_ = EnergySplit;
    }else {
      throw cms::Exception("HGCTriggerParameterError")
        << "Unknown cluster association strategy'" << cluster_association_strategy_;
    } 

    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR: " << dr_
                                                <<"\nMulticluster minimum transverse-momentum: " << ptC3dThreshold_ ;

    id_ = std::unique_ptr<HGCalTriggerClusterIdentificationBase>{ HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") };
    id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));

}



float HGCalHistoClusteringImpl::dR( const l1t::HGCalCluster & clu,
                                         const GlobalPoint & seed) const
{

    Basic3DVector<float> seed_3dv( seed );
    GlobalPoint seed_proj( seed_3dv / seed.z() );
    return (seed_proj - clu.centreProj() ).mag();

}



std::vector<l1t::HGCalMulticluster> HGCalHistoClusteringImpl::clusterSeedMulticluster(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
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




void HGCalHistoClusteringImpl::clusterizeHisto( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
                                                     const std::vector<std::pair<GlobalPoint, double> > & seedPositionsEnergy,
                                                     const HGCalTriggerGeometryBase & triggerGeometry,
                                                     l1t::HGCalMulticlusterBxCollection & multiclusters)
{


    /* clusterize clusters around seeds */
    std::vector<l1t::HGCalMulticluster> multiclustersTmp = clusterSeedMulticluster(clustersPtrs,seedPositionsEnergy);    
    /* making the collection of multiclusters */
    finalizeClusters(multiclustersTmp, multiclusters, triggerGeometry);

}





void
HGCalHistoClusteringImpl::
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
