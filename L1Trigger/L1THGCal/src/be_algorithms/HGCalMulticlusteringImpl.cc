

#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    multiclusterAlgoType_(conf.getParameter<string>("type_multicluster")),
    distDbscan_(conf.getParameter<double>("dist_dbscan_multicluster")),
    minNDbscan_(conf.getParameter<unsigned>("minN_dbscan_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster DBSCAN Clustering distance: " << distDbscan_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster clustering min number of subclusters: " << minNDbscan_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster type of multiclustering algortihm: " << multiclusterAlgoType_;
    id_.reset( HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") );
    id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification")); 
}


bool HGCalMulticlusteringImpl::isPertinent( const l1t::HGCalCluster & clu, 
                                            const l1t::HGCalMulticluster & mclu, 
                                            double dR ) const
{
    HGCalDetId cluDetId( clu.detId() );
    HGCalDetId firstClusterDetId( mclu.detId() );
    
    if( cluDetId.zside() != firstClusterDetId.zside() ){
        return false;
    }
    if( ( mclu.centreProj() - clu.centreProj() ).mag() < dR ){
        return true;
    }
    return false;

}


void HGCalMulticlusteringImpl::findNeighbor( const std::vector<std::pair<unsigned int,double>>&  rankedList,
                                             unsigned int searchInd,
                                             const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs, 
                                             std::vector<unsigned int>& neighbors
                                            ){

    if(clustersPtrs.size() <= searchInd || clustersPtrs.size() < rankedList.size()){
        throw cms::Exception("IndexOutOfBound: clustersPtrs in 'findNeighbor'");
    }

    for(unsigned int ind = searchInd+1; ind < rankedList.size() && fabs(rankedList.at(ind).second - rankedList.at(searchInd).second) < distDbscan_ ; ind++){

        if(clustersPtrs.size() <= rankedList.at(ind).first){
            throw cms::Exception("IndexOutOfBound: clustersPtrs in 'findNeighbor'");

        } else if(((*(clustersPtrs[rankedList.at(ind).first])).centreProj() - (*(clustersPtrs[rankedList.at(searchInd).first])).centreProj()).mag() < distDbscan_){
            neighbors.push_back(ind);
        }
    }

    for(unsigned int ind = 0; ind < searchInd && fabs(rankedList.at(searchInd).second - rankedList.at(ind).second) < distDbscan_ ; ind++){

        if(clustersPtrs.size() <= rankedList.at(ind).first){
            throw cms::Exception("IndexOutOfBound: clustersPtrs in 'findNeighbor'");

        } else if(((*(clustersPtrs[rankedList.at(ind).first])).centreProj() - (*(clustersPtrs[rankedList.at(searchInd).first])).centreProj()).mag() < distDbscan_){
            neighbors.push_back(ind);
        }
    }
}


void HGCalMulticlusteringImpl::clusterizeDR( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters,
                                           const HGCalTriggerGeometryBase & triggerGeometry)
{

    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    int iclu = 0;
    for(std::vector<edm::Ptr<l1t::HGCalCluster>>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){

        double minDist = dr_;
        int targetMulticlu = -1;

        for(unsigned imclu=0; imclu<multiclustersTmp.size(); imclu++){

            if(!this->isPertinent(**clu, multiclustersTmp.at(imclu), dr_)) continue;

            double d = ( multiclustersTmp.at(imclu).centreProj() - (*clu)->centreProj() ).mag() ;
            if(d<minDist){
                minDist = d;
                targetMulticlu = int(imclu);
            }
        }

        if(targetMulticlu<0) multiclustersTmp.emplace_back( *clu );
        else multiclustersTmp.at( targetMulticlu ).addConstituent( *clu );

    }

    /* making the collection of multiclusters */
    finalizeClusters(multiclustersTmp, multiclusters, triggerGeometry);
    
}
void HGCalMulticlusteringImpl::clusterizeDBSCAN( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs, 
                                                 l1t::HGCalMulticlusterBxCollection & multiclusters,
                                                 const HGCalTriggerGeometryBase & triggerGeometry)
{

    std::vector<l1t::HGCalMulticluster> multiclustersTmp;
    l1t::HGCalMulticluster mcluTmp;
    std::vector<bool> visited(clustersPtrs.size(),false);
    std::vector<bool> merged (clustersPtrs.size(),false);
    std::vector<std::pair<unsigned int,double>>  rankedList;
    rankedList.reserve(clustersPtrs.size());
    std::vector<std::vector<unsigned int>> neighborList;
    neighborList.reserve(clustersPtrs.size());

    int iclu = 0, imclu = 0, neighNo;
    double dist = 0.;

    for(std::vector<edm::Ptr<l1t::HGCalCluster>>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
        dist = (*clu)->centreProj().mag()*HGCalDetId((*clu)->detId()).zside();
        rankedList.push_back(std::make_pair(iclu,dist));
    }  
    iclu = 0;
    std::sort(rankedList.begin(), rankedList.end(), [](auto &left, auto &right) {
            return left.second < right.second;
            });

    for(const auto& cluRanked: rankedList){
        std::vector<unsigned int> neighbors;      

        if(!visited.at(iclu)){
            visited.at(iclu) = true;
            findNeighbor(rankedList, iclu, clustersPtrs, neighbors);
            neighborList.push_back(std::move(neighbors));

            if(neighborList.at(iclu).size() >= minNDbscan_) {
                multiclustersTmp.emplace_back( clustersPtrs[cluRanked.first] );
                merged.at(iclu) = true;
                /* dynamic range loop: range-based loop syntax cannot be employed */
                for(unsigned int neighInd = 0; neighInd < neighborList.at(iclu).size(); neighInd++){
                    neighNo = neighborList.at(iclu).at(neighInd);
                    /* This condition also ensures merging of clusters visited by other clusters but not merged. */
                    if(!merged.at(neighNo) ){
                        merged.at(neighNo) = true;          
                        multiclustersTmp.at(imclu).addConstituent( clustersPtrs[rankedList.at(neighNo).first] );

                        if(!visited.at(neighNo)){
                            visited.at(neighNo) = true;
                            std::vector<unsigned int> secNeighbors;
                            findNeighbor(rankedList, neighNo,clustersPtrs, secNeighbors);

                            if(secNeighbors.size() >= minNDbscan_){
                                neighborList.at(iclu).insert(neighborList.at(iclu).end(), secNeighbors.begin(), secNeighbors.end());
                            }
                        }
                    }
                }
                imclu++;
            }
        }
        else neighborList.push_back(std::move(neighbors));
        iclu++;    
    }
    /* making the collection of multiclusters */
    finalizeClusters(multiclustersTmp, multiclusters, triggerGeometry);
}


void
HGCalMulticlusteringImpl::
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
