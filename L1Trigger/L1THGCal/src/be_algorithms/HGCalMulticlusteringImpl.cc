#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    calibSF_(conf.getParameter<double>("calibSF_multicluster")),
    multiclusterAlgoType_(conf.getParameter<string>("type_multicluster")),
    distDbscan_(conf.getParameter<double>("dist_dbscan_multicluster")),
    minNDbscan_(conf.getParameter<unsigned>("minN_dbscan_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster global calibration factor: " << calibSF_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster DBSCAN Clustering distance: " << distDbscan_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster clustering min number of subclusters: " << minNDbscan_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster type of multiclustering algortihm: " << multiclusterAlgoType_;
    
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


bool HGCalMulticlusteringImpl::isNeighbor( const l1t::HGCalCluster & clu1, 
                                            const l1t::HGCalCluster & clu2) const
{
  HGCalDetId cluDetId( clu2.detId() );
  HGCalDetId firstClusterDetId( clu1.detId() );
  
  if( cluDetId.zside() != firstClusterDetId.zside() ){
    return false;
  }
  double dist = (clu1.centreProj() - clu2.centreProj() ).mag();
  
  if(dist < distDbscan_ ) {
    return true;
  }
  return false;
}


void HGCalMulticlusteringImpl::findNeighbor( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
					     const l1t::HGCalCluster & cluster,
					     std::vector<int> & neighbors
					     )
{
  int iclu = 0;

  for(edm::PtrVector<l1t::HGCalCluster>::iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
    
    if(isNeighbor(cluster, **clu)){
      neighbors.push_back(iclu);
    }
  }
}

void HGCalMulticlusteringImpl::clusterizeDR( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters)
{
           
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    int iclu = 0;
    for(edm::PtrVector<l1t::HGCalCluster>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
        
        int imclu=0;
        vector<int> tcPertinentMulticlusters;
        for( const auto& mclu : multiclustersTmp ){
            if( this->isPertinent(**clu, mclu, dr_) ){
                tcPertinentMulticlusters.push_back(imclu);
            }
            ++imclu;
        }
        if( tcPertinentMulticlusters.size() == 0 ){
            multiclustersTmp.emplace_back( *clu );
        }
        else{
            unsigned minDist = 1;
            unsigned targetMulticlu = 0; 
            for( int imclu : tcPertinentMulticlusters ){
                double d = ( multiclustersTmp.at(imclu).centreProj() - (*clu)->centreProj() ).mag() ;
                if( d < minDist ){
                    minDist = d;
                    targetMulticlu = imclu;
                }
            } 

            multiclustersTmp.at( targetMulticlu ).addConstituent( *clu );
            
        }        
    }

    /* making the collection of multiclusters */
    for( unsigned i(0); i<multiclustersTmp.size(); ++i ){
        math::PtEtaPhiMLorentzVector calibP4(  multiclustersTmp.at(i).pt() * calibSF_, 
                                               multiclustersTmp.at(i).eta(), 
                                               multiclustersTmp.at(i).phi(), 
                                               0. );
        // overwriting the 4p with the calibrated 4p     
        multiclustersTmp.at(i).setP4( calibP4 );
        
        if( multiclustersTmp.at(i).pt() > ptC3dThreshold_ ){

            //compute shower shape
            multiclustersTmp.at(i).set_showerLength(shape_.showerLength(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_firstLayer(shape_.firstLayer(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_sigmaEtaEtaTot(shape_.sigmaEtaEtaTot(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_sigmaEtaEtaMax(shape_.sigmaEtaEtaMax(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_sigmaPhiPhiTot(shape_.sigmaPhiPhiTot(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_sigmaPhiPhiMax(shape_.sigmaPhiPhiMax(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_sigmaZZ(shape_.sigmaZZ(multiclustersTmp.at(i)));
            multiclustersTmp.at(i).set_eMax(shape_.eMax(multiclustersTmp.at(i)));

            multiclusters.push_back( 0, multiclustersTmp.at(i));  
        }
    }
    
}
void HGCalMulticlusteringImpl::clusterizeDBSCAN( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters)
{

  std::vector<l1t::HGCalMulticluster> multiclustersTmp;
  l1t::HGCalMulticluster mcluTmp;
  std::vector<bool> visited(clustersPtrs.size(),false);
  std::vector<bool> merged (clustersPtrs.size(),false);
  std::vector<std::vector<int>> neighborList;
  int iclu = 0, imclu = 0;

  for(edm::PtrVector<l1t::HGCalCluster>::iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
      std::vector<int> neighbors;      
      
      if(!visited.at(iclu)){
	visited.at(iclu)=true;
	findNeighbor(clustersPtrs, **clu, neighbors);
	neighborList.push_back(neighbors);

	if(neighborList.at(iclu).size() > minNDbscan_) {
	  multiclustersTmp.emplace_back( *clu );
	  merged.at(iclu) = true;
	  
	  for(unsigned int neighNo = 0; neighNo < neighborList.at(iclu).size(); neighNo++){
	    if(!visited.at(neighborList.at(iclu).at(neighNo))){
	      visited.at(neighborList.at(iclu).at(neighNo)) = true;
	      std::vector<int> secNeighbors;
	      findNeighbor(clustersPtrs,*(clustersPtrs[neighborList.at(iclu).at(neighNo)]), secNeighbors);
	      multiclustersTmp.at(imclu).addConstituent( clustersPtrs[neighborList.at(iclu).at(neighNo)]);
	      merged.at(neighborList.at(iclu).at(neighNo)) = true;
	      
	      if(secNeighbors.size() > minNDbscan_){
		neighborList.at(iclu).insert(neighborList.at(iclu).end(), secNeighbors.begin(), secNeighbors.end());
	      }
	      
	    } else if(!merged.at(neighborList.at(iclu).at(neighNo)) ){
	      merged.at(neighborList.at(iclu).at(neighNo)) = true;		
	      multiclustersTmp.at(imclu).addConstituent( clustersPtrs[neighborList.at(iclu).at(neighNo)] );
	      }
	    }
	    imclu++;
	}
      }
      else neighborList.push_back(neighbors);
  }
  
  /* making the collection of multiclusters */
  for( unsigned i(0); i<multiclustersTmp.size(); ++i ){
    math::PtEtaPhiMLorentzVector calibP4(  multiclustersTmp.at(i).pt() * calibSF_, 
					   multiclustersTmp.at(i).eta(), 
					   multiclustersTmp.at(i).phi(), 
					   0. );
    // overwriting the 4p with the calibrated 4p     
    multiclustersTmp.at(i).setP4( calibP4 );
    
    if( multiclustersTmp.at(i).pt() > ptC3dThreshold_ ){
      
      //compute shower shape
      multiclustersTmp.at(i).set_showerLength(shape_.showerLength(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_firstLayer(shape_.firstLayer(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_sigmaEtaEtaTot(shape_.sigmaEtaEtaTot(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_sigmaEtaEtaMax(shape_.sigmaEtaEtaMax(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_sigmaPhiPhiTot(shape_.sigmaPhiPhiTot(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_sigmaPhiPhiMax(shape_.sigmaPhiPhiMax(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_sigmaZZ(shape_.sigmaZZ(multiclustersTmp.at(i)));
      multiclustersTmp.at(i).set_eMax(shape_.eMax(multiclustersTmp.at(i)));
      
      multiclusters.push_back( 0, multiclustersTmp.at(i));  
    }
  }
}
