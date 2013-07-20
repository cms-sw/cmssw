#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterInit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdexcept>
#include <string>
#include <sstream>
#include <map>


using namespace std;
using namespace reco;

unsigned PFHcalSuperClusterAlgo::prodNum_ = 1;

//for debug only 
//#define PFLOW_DEBUG

PFHcalSuperClusterAlgo::PFHcalSuperClusterAlgo() :
  pfClusters_( new vector<reco::PFCluster> ),
  pfSuperClusters_( new vector<reco::PFSuperCluster> ),
  pfClustersHO_( new vector<reco::PFCluster> ),
  pfSuperClustersHO_( new vector<reco::PFSuperCluster>),
  debug_(false) 
{

}

void
PFHcalSuperClusterAlgo::write() {
}

//initialize clutsering of clusters into superclusters
void PFHcalSuperClusterAlgo::doClustering( const PFClusterHandle& clustersHandle, const PFClusterHandle& clustersHOHandle ) {
  const reco::PFClusterCollection& clusters = * clustersHandle;
  const reco::PFClusterCollection& clustersHO = * clustersHOHandle;

  // cache the Handle to the clusters
  clustersHandle_ = clustersHandle;
  clustersHOHandle_ = clustersHOHandle;

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}


void PFHcalSuperClusterAlgo::doClustering( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO ) {
  // using clusters without a Handle, clear to avoid a stale member
  clustersHandle_.clear();
  clustersHOHandle_.clear();

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}


// calculate cluster position based on Rec Hits and Fracs: Rachel Myers, July 2012 (updated by Josh Kaisen April 2013)
std::pair<double, double> PFHcalSuperClusterAlgo::calcPosition(const reco::PFCluster& cluster)
{
  double numeratorEta = 0.0;
  double numeratorPhi = 0.0;
  double denominator = 0.0;
  double posEta = 0.0;
  double posPhi = 0.0;
  //Initial weight set by Chris Tully
  double w0_ = 4.2;

  const double clusterEnergy = cluster.energy();
  if (cluster.energy()>0.0 && cluster.recHitFractions().size()>0) {
    const std::vector <reco::PFRecHitFraction >& pfhitsandfracs = cluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {//Calculate position from RecHits and scale based on energy of hit
      const reco::PFRecHitRef rechit = it->recHitRef();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      //Confine Phi to -pi to +pi angular range
      while (hitPhi > +Geom::pi()) { hitPhi -= Geom::twoPi(); }
      while (hitPhi < -Geom::pi()) { hitPhi += Geom::twoPi(); }
      double hitEnergy = rechit->energy();
      //Scale based on hit energy
      const double w = std::max(0.0, w0_ + log(hitEnergy / clusterEnergy));
      denominator += w;//Keep a running total of the weights for renormaliztion
      numeratorEta += w*hitEta;//Calculate positions based on the energy contained in the rechit
      numeratorPhi += w*hitPhi;
    }
    if(denominator == 0)
      posEta = posPhi = 0.0;//If the denominator is somehow 0, set positions to 0, this isn't necessarily good though because 0 is still a  valid position for both.
    else{
      posEta = numeratorEta/denominator;//Renormalize
      posPhi = numeratorPhi/denominator;
    }
  }
  
 
  pair<double, double> EtaPhi(posEta,posPhi);//Report eta and phi
  return EtaPhi;
}


//calculates the range a cluster has to be merged into a supercluster based on the distribution of the rec hits and fracs
std::pair<double, double> PFHcalSuperClusterAlgo::calcWidth(const reco::PFCluster& cluster, std::pair<double,double> Position)
{
  double numeratorEtaEta = 0;
  double numeratorPhiPhi = 0;
  double widthEta = 0.0;
  double widthPhi = 0.0;
  double denominator = 0.0;
  //Initial weight set by Chris Tully
  double w0_ = 4.2;

  //Get position from included pair
  double clusterEta = Position.first;
  double clusterPhi = Position.second;
    
  const double clusterEnergy = cluster.energy();
  if (cluster.energy()>0.0 && cluster.recHitFractions().size()>0) {
    const std::vector <reco::PFRecHitFraction >& pfhitsandfracs = cluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      double dPhi, dEta;
      const reco::PFRecHitRef rechit = it->recHitRef();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      //Confine Phi to -pi to +pi angular range
      while (hitPhi > +Geom::pi()) { hitPhi -= Geom::twoPi(); }
      while (hitPhi < -Geom::pi()) { hitPhi += Geom::twoPi(); }
      double hitEnergy = rechit->energy();
      
      
      dEta  = hitEta - clusterEta;
      dPhi  = hitPhi - clusterPhi;
      
      const double w = std::max(0.0, w0_ + log(hitEnergy / clusterEnergy));
      denominator += w;
      
      numeratorEtaEta += w * dEta * dEta;//Sum up deviations from the mean squared so that they may be added in quadriture
      //      numeratorEtaPhi += w * dEta * dPhi;
      numeratorPhiPhi += w * dPhi * dPhi;
    }
    if(denominator == 0)
      widthEta = widthPhi = 0.0;//Set width to zero if there is no energy in the cluster

    else{
      double covEtaEta = numeratorEtaEta / denominator;//Renomalize
      //    double covEtaPhi_ = numeratorEtaPhi / denominator;
      double covPhiPhi = numeratorPhiPhi / denominator;
      widthEta = sqrt(std::abs(covEtaEta));//make the width the addition in quadriture of the addition of all the rechits deivation from teh mean
      widthPhi = sqrt(std::abs(covPhiPhi));
    }
  }
  pair<double, double> EtaPhi(widthEta, widthPhi);
  return EtaPhi;
}

//Test to see if clusters are close enough to be merged, can be changed to output a score based on how accurately two clusters could be merged, to be comapred with other clusters that might merge with the same one. Josh Kaisen April 2013
std::pair<bool , double> PFHcalSuperClusterAlgo::TestMerger( std::pair<double,double> Position1, std::pair<double,double> Width1, std::pair<double,double> Position2, std::pair<double,double> Width2 ){

  double hcaleta1=0.0;
  double hcalphi1=0.0;
  double hcaleta2=0.0;
  double hcalphi2=0.0;
  double dR = 0.0;
  double dEta = 0.0;
  double dPhi = 0.0;
  //Cut constants hardcoded by Chris Tully
  double dRcut=0.17;
  double dEtacut = 0.0;
  double dPhicut = 0.0; 
  double etaScale = 1.0;
  double phiScale = 0.5;   

  double w1 = Width1.first;
  if (w1 == 0) w1 = 0.087;
  double w3 = Width1.second;
  if (w3 < 0.087) w3 = 0.087;
  hcaleta1 = Position1.first;
  hcalphi1 = Position1.second;
  hcaleta2 = Position2.first;
  hcalphi2 = Position2.second;
  dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
  dEta = std::abs(hcaleta1 - hcaleta2);
  dPhi = std::abs(deltaPhi(hcalphi1, hcalphi2));	 
  double w2 = Width2.first;
  if (w2 == 0) w2 = 0.087;
  double w4 = Width2.second;
  if (w4 < 0.087) w4 = 0.087;
  double etawidth = sqrt(w1*w1 + w2*w2);
  double phiwidth = sqrt(w3*w3 + w4*w4);
  dEtacut = etaScale*etawidth;
  dPhicut = phiScale*phiwidth;
  //This keeps intact the formulation before my editing, where there were only up to five layers for the the cluster to work with and some special behavior with the 3rd.
  //            cout << " depth 1-2 dR = " << dR <<endl;
  std::pair<bool,double> PosInfo( ((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)), dR );//Create pair with bool to indicate whether the two clusters are in range, and the range
  //return((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut));//Say whether the clusers fall within each other's cuts
  return(PosInfo);
}

//Forms links to indicate that one cluster of large depth should bond with another cluster of smaller depth, based on delta R in this iteration. by Josh Kaisen
void PFHcalSuperClusterAlgo::FormLinks( std::vector< bool >& lmerge, std::vector<unsigned>& iroot, std::vector<double>& idR, unsigned d, unsigned ic,  std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepth, std::pair<double,double> Position, std::pair<double,double> Width ){
  //iterative process will stop when there are no longer any clusters in the depths
  for (unsigned d1=1; d1<3; d1++)
    for (unsigned ic1=0; ic1<clustersbydepth[d+d1].size();++ic1){//loop over clusters for current depth d+d1
      std::pair<double,double> Position1 = calcPosition(clustersbydepth[d+d1][ic1].first);//Get position
      std::pair<double,double> Width1 = calcWidth(clustersbydepth[d+d1][ic1].first, Position1);//Get Width
      std::pair<bool,double> Range1 = TestMerger(Position,Width,Position1,Width1);//Get Range
      if(Range1.first){//If the two clusters are in range then begin linking
	//std::cout<<"passed Range cut" << std::endl;
	//std::cout << "lmerge: " << lmerge[clustersbydepth[d+d1][ic1].second] << std::endl;
	if(lmerge[clustersbydepth[d+d1][ic1].second]){//If the second cluster has been linked before, test proximity
	  if(Range1.second < idR[clustersbydepth[d+d1][ic1].second]){
	    iroot[clustersbydepth[d+d1][ic1].second] = clustersbydepth[d][ic].second;//If closer proximity change root
	    idR[clustersbydepth[d+d1][ic1].second] = Range1.second;//adjust delta R info
	  }
	}
	else {//Go to see what else matches with this cluster ( this will have already been done if it was merged before so all the links will already be set)
	  lmerge[clustersbydepth[d+d1][ic1].second]=true;//Change linked info to true
	  //std::cout << " clustersbydepth[d+d1][ic1].second : " << clustersbydepth[d+d1][ic1].second << std::endl;
	  idR[clustersbydepth[d+d1][ic1].second] = Range1.second;//set delta R
	  iroot[clustersbydepth[d+d1][ic1].second] = clustersbydepth[d][ic].second;//set root link index
	  FormLinks( lmerge, iroot, idR, (d+d1), ic1, clustersbydepth, Position1, Width1);
	}
      }
    }
}
 
//Merges clusters as indicated by the root information recorded in FormLinks, will also indicate if there is an HO cluster in the supercluster. by Josh Kaisen
void PFHcalSuperClusterAlgo::MergeClusters( std::vector< bool >& lmerge, std::vector<unsigned>& iroot, std::vector< bool >& lmergeHO, std::vector<unsigned>& irootHO, unsigned depthHO, unsigned d, unsigned ic, std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepth, std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepthHO, edm::PtrVector< reco::PFCluster >& mergeclusters, bool HOSupercluster ){
  
  // if( !((depthHO - d) > 2 ) ){//Check HO based on some condition
  //   std::cout << " 111111111111111111111111111111111111111111" << std::endl;
  //   for(unsigned icHO=0; icHO<clustersbydepthHO[depthHO].size(); icHO++)//Check if the HO clusters will be getting merged
  //     if(lmergeHO[clustersbydepthHO[depthHO][icHO].second] && irootHO[clustersbydepthHO[depthHO][icHO].second] == clustersbydepth[d][ic].second){
  // 	std::cout << " HO index: " << clustersbydepthHO[depthHO][icHO].second << std::endl;
  // 	mergeclusters.push_back(PFClusterPtr( clustersHOHandle_, clustersbydepthHO[depthHO][icHO].second ));
  // 	std::cout << "333333333333333333333333333333333333333333" << std::endl;
  // 	HOSupercluster = true;
  //     }
  // }
  for (unsigned d1=1;d1<3; d1++)//Go through remaining clusters and see what else will be merged
    for (unsigned ic1=0; ic1<clustersbydepth[d+d1].size();++ic1)
      if(lmerge[clustersbydepth[d+d1][ic1].second] && iroot[clustersbydepth[d+d1][ic1].second] == clustersbydepth[d][ic].second){
	mergeclusters.push_back(PFClusterPtr( clustersHandle_, clustersbydepth[d+d1][ic1].second));
	MergeClusters( lmerge, iroot, lmergeHO, irootHO, depthHO, (d+d1), ic1, clustersbydepth, clustersbydepthHO, mergeclusters, HOSupercluster);
      }
  
}
// do clustering with new widths, positions, parameters, merging conditions: Rachel Myers, July 2012 (Updated by Josh Kaisen, April 2013)
void PFHcalSuperClusterAlgo::doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO) {

  //  double dRcut=0.30;

  //initilize data structures
  if ( pfClusters_.get() )
    pfClusters_->clear();
  else 
    pfClusters_.reset( new std::vector<reco::PFCluster> );

  if ( pfSuperClusters_.get() )
    pfSuperClusters_->clear();
  else 
    pfSuperClusters_.reset( new std::vector<reco::PFSuperCluster> );

  // compute cluster depth index
  
  //  create a map giving the clusters in each depth and keeping the infomation of their position in the handle. 
  std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > > clustersbydepth;//in order each portion is: map<depth , vector<pair< Cluster, Order in handle > > > 
  std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > > clustersbydepthHO;//Meant for iteration over depths and report based on the initial handle
  

  if(debug_) std::cout << " Finding depths from recfractions " << std::endl;
  int mclustersHB=0;
  int mclustersHE=0;
  unsigned MAXdepthHBHE=0;//!!!Get the maximum depth of hits in the HB in order to look at HO behavior
  //Record depths of clusters based on rechits and generate map based on depth
  for (unsigned short ic=0; ic<clusters.size();++ic)
    {

      if( clusters[ic].layer() == PFLayer::HCAL_BARREL1
	  || clusters[ic].layer() == PFLayer::HCAL_ENDCAP ) { //Hcal case
	if( clusters[ic].layer() == PFLayer::HCAL_BARREL1) mclustersHB++;//!!!No purpose to these currently.. Josh
	if( clusters[ic].layer() == PFLayer::HCAL_ENDCAP) mclustersHE++;
 
	const reco::PFRecHitRef& RecHit = (clusters[ic].recHitFractions().begin())->recHitRef();
	unsigned depth = ((HcalDetId)(RecHit->detId())).depth();
	clustersbydepth[depth].push_back(std::make_pair(clusters[ic],ic));//add to the map at the depth of the cluster in the loop and continue loop over all clusters
	if(debug_) std::cout << "HcalDetId: " << RecHit->detId() << std::endl;//<< " SubDetId: " << RecHit->subdetId() << std::endl;
	if(debug_) std::cout << " HBHE Depth: " << depth << std::endl;
	if(depth > MAXdepthHBHE)
	  MAXdepthHBHE = depth;
      }
    }
  
  
  unsigned depthHO = 0;
  //Do the same for HO clusters
  for (unsigned short ic=0; ic<clustersHO.size();++ic)
    {
      if( clustersHO[ic].layer() == PFLayer::HCAL_BARREL2) { //HO case
	const reco::PFRecHitRef& RecHit = (clustersHO[ic].recHitFractions().begin())->recHitRef();
	unsigned depth = ((HcalDetId)(RecHit->detId())).depth();
	clustersbydepthHO[depth].push_back(std::make_pair(clustersHO[ic],ic));
	depthHO = depth;
	if(debug_) std::cout << " HO Depth: " << depth << std::endl;
      }    
    }

  
  std::vector<unsigned> iroot(clusters.size());//Tells a cluster which cluster it is to be merged with
  std::vector<double> idR(clusters.size());//Stores the delta R information for a cluster's link once it has been forged
  std::vector<unsigned> irootHO(clustersHO.size());
  std::vector<double> idRHO(clustersHO.size());
  std::vector< bool > lmerge(clusters.size());//indicates whether a cluster has already been linked
  std::vector< bool > lmergeHO(clustersHO.size());


  if(debug_) std::cout<<"  filling bools with false " << std::endl;
  std::fill(lmerge.begin(), lmerge.end(), false);//All cluster begin unlinked
  std::fill(lmergeHO.begin(), lmergeHO.end(), false);

  if(debug_) std::cout << "    Forming links " << std::endl;
  //Will loop through clusters and determine which are linked, linking is down with priority based on delta R
  for (unsigned d=0; d<MAXdepthHBHE;++d)
    for(unsigned ic=0; ic<clustersbydepth[d].size(); ic++)//loop over clusters for current depth d
      if(!lmerge[clustersbydepth[d][ic].second]){//If the cluster wasn't linked in any other super cluster then it may begin its own
	std::pair<double,double> Position = calcPosition(clustersbydepth[d][ic].first);//Get position of seed cluster
	std::pair<double,double> Width = calcWidth(clustersbydepth[d][ic].first, Position);//Get width of seed cluster
	//std::cout << " Forming links Inside code" << std::endl;
	FormLinks( lmerge, iroot, idR, d, ic, clustersbydepth, Position, Width );//Form links with across other depths except HO
      }//end loop over clusters
  //end loop over depths

  
  //Loop over the end layers of the HB and assign HO info
  //if ( !((depthHO - MAXdepthHBHE) > 2 ))//Need to change this to detect maximum HB layer through geo.. somehow
  if(debug_) std::cout << "     Forming links with HO " << std::endl;
  for(unsigned ic=0; ic<clustersbydepthHO[depthHO].size(); ic++){
    std::pair<double,double> PositionHO = calcPosition(clustersbydepthHO[depthHO][ic].first);
    std::pair<double,double> WidthHO = calcWidth(clustersbydepthHO[depthHO][ic].first, PositionHO);   
    for(unsigned ic1=0; ic1<clustersbydepth[3].size(); ic1++){//Form links for the HO layers ( currently assumed to be 1 layer )
      std::pair<double,double> Position = calcPosition(clustersbydepth[3][ic1].first);
      std::pair<double,double> Width = calcWidth(clustersbydepth[3][ic1].first, Position);
      std::pair<bool,double> Range = TestMerger(Position,Width,PositionHO,WidthHO);
      if(Range.first){
	if(lmergeHO[clustersbydepthHO[depthHO][ic].second]){
	  if(Range.second < idRHO[clustersbydepthHO[depthHO][ic].second]){
	    irootHO[clustersbydepthHO[depthHO][ic].second] = clustersbydepth[3][ic1].second;
	    idRHO[clustersbydepthHO[depthHO][ic].second] = Range.second;
	  }
	}
	else{
	  irootHO[clustersbydepthHO[depthHO][ic].second] = clustersbydepth[3][ic1].second;
	  idRHO[clustersbydepthHO[depthHO][ic].second] = Range.second;
	  lmergeHO[clustersbydepthHO[depthHO][ic].second]=true;
	}
      }
    }
  }
  // for( unsigned i = 0; i<lmerge.size(); i++)
  //   if(lmerge[i]) std::cout << "lmerge[" << i << "]: " << lmerge[i] << std::endl;
  // Have links up to here.
  if(debug_) std::cout << " Forming superclusters " << std::endl;
  edm::PtrVector< reco::PFCluster >  mergeclusters;//List of all clusters to be merged in a supercluster creation
  bool HOSupercluster = false;//Does the current mergeclusters contain an HO cluster
  for (unsigned d=0; d<=MAXdepthHBHE;++d) //loop over depths 1 - 5 //depth min 1 //depth max 5
    for(unsigned ic=0; ic<clustersbydepth[d].size(); ic++){//loop over clusters for current depth d
      
      if(!lmerge[clustersbydepth[d][ic].second]){//if this is a seed cluster (i.e. wasn't linked)
	mergeclusters.push_back(PFClusterPtr( clustersHandle_, clustersbydepth[d][ic].second));//start the supercluster
	//std::cout << "111111111111111111111111111111111111111111111111" << std::endl;
	MergeClusters( lmerge, iroot, lmergeHO, irootHO, depthHO, d, ic, clustersbydepth, clustersbydepthHO, mergeclusters, HOSupercluster );//Check merger across all other depth including HO
	//std::cout << "22222222222222222222222222222222222222222222222" << std::endl;
	if(mergeclusters.size()>0) {//Creates superclusters based on mergecluster (Can have a supercluster of 1 cluster)
	    //            cout << " number of clusters to merge: " <<mergeclusters.size()<<endl;
	  reco::PFSuperCluster ipfsupercluster(mergeclusters);//initialize the supercluster
	  PFHcalSuperClusterInit init;//Create the super cluster intiliazation object
	  //std::cout << " mergeclusters size: " << mergeclusters.size() << std::endl;
	  init.initialize( ipfsupercluster, mergeclusters);//will create superclusters based on the information in mergeclusters
	  //std::cout << "3333333333333333333333333333333333333333333333" << std::endl;
	  if(HOSupercluster){//If there was an HO cluster in the supercluster then it gets recorded seperately as an HO supercluster
	    
	    pfSuperClustersHO_->push_back(ipfsupercluster);
	    pfClustersHO_->push_back((reco::PFCluster)ipfsupercluster);
	  }
	  else{//Otherwise it is reported as a normal cluster
	    
	    pfSuperClusters_->push_back(ipfsupercluster);//Add supercluster to list of superclusters
	    pfClusters_->push_back((reco::PFCluster)ipfsupercluster);//Add supercluster to list of clusters, !!!Note sure how depth info is stored here
	  }
	  //std::cout << "444444444444444444444444444444444444444444444" << std::endl;
	  
	  mergeclusters.clear();//clear current buffer
	  HOSupercluster = false;//reset HO truth indicator
	}
      }
    }

  edm::PtrVector< reco::PFCluster >  mergeclustersHO;
  if(debug_) std::cout << " Merging HOSuperClusters " << std::endl;
  //merge the rest of the HO clustesrs into superclusters
  for(unsigned icHO=0; icHO<clustersbydepthHO[depthHO].size(); icHO++)
    if(!lmergeHO[clustersbydepthHO[depthHO][icHO].second])
      {
  	
  	
  	mergeclustersHO.push_back( PFClusterPtr( clustersHOHandle_, clustersbydepthHO[depthHO][icHO].second ) );
  	
  	reco::PFSuperCluster ipfsuperclusterHO( mergeclustersHO );
  	PFHcalSuperClusterInit init;
  	
  	init.initialize ( ipfsuperclusterHO, mergeclustersHO );
	
        pfSuperClustersHO_->push_back(ipfsuperclusterHO);
	pfClustersHO_->push_back((reco::PFCluster)ipfsuperclusterHO);
	

  	mergeclustersHO.clear();
      }
        

  if(debug_) std::cout << " Exiting SuperClusterAlgo " << std::endl;
  //Clean data structs used
  clustersbydepth.clear();
  clustersbydepthHO.clear();

  lmerge.clear();
  lmergeHO.clear();
  mergeclusters.clear();
  mergeclustersHO.clear();
  iroot.clear();
  idR.clear();
  irootHO.clear();
  idRHO.clear();
  
  
}

ostream& operator<<(ostream& out,const PFHcalSuperClusterAlgo& algo) { 
  if(!out) return out;
  out<<"PFSuperClusterAlgo parameters : "<<endl;
  out<<"-----------------------------------------------------"<<endl;
  
  out<<endl;
  out<<algo.pfClusters_->size()<<" clusters:"<<endl;
  
  for(unsigned i=0; i<algo.pfClusters_->size(); i++) {
    out<<(*algo.pfClusters_)[i]<<endl;
    
    if(!out) return out;
  }
  
  out<<algo.pfSuperClusters_->size()<<" superclusters:"<<endl;
    
  for(unsigned i=0; i<algo.pfSuperClusters_->size(); i++) {
    out<<(*algo.pfSuperClusters_)[i]<<endl;
    
    if(!out) return out;
  }   
  return out;
}
