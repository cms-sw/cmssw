#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"


void HGCalImagingAlgo::populate(const HGCRecHitCollection& hits){
  //loop over all hits and create the Hexel structure, skip energies below ecut
  computeThreshold();

  for (unsigned int i=0;i<hits.size();++i) {
    const HGCRecHit& hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layer = rhtools_.getLayerWithOffset(detid);
    float thickness = -9999.;
    float sigmaNoise = -9999.;
    if(dependSensor){
      if (layer<= lastLayerFH)
	thickness = rhtools_.getSiThickness(detid);
      HGCalDetId hgDetId(detid);
      double storedThreshold=thresholds[layer-1][layer<=lastLayerFH ? hgDetId.wafer() : 0];
      if(hgrh.energy() <  storedThreshold) continue; //this sets the ZS threshold at ecut times the sigma noise for the sensor
    }
    if(!dependSensor && hgrh.energy() < ecut) continue;

    layer += int(HGCalDetId(detid).zside()>0)*(maxlayer+1);

    // determine whether this is a half-hexagon
    bool isHalf = rhtools_.isHalfCell(detid);
    const GlobalPoint position( std::move( rhtools_.getPosition( detid ) ) );
    if(zees[layer]==0) {
      zees[layer] = position.z();
    }
    //here's were the KDNode is passed its dims arguments - note that these are *copied* from the Hexel
    if(thickness<0.)thickness = 0.;
    points[layer].emplace_back(Hexel(hgrh,detid,isHalf,sigmaNoise,thickness,&rhtools_),position.x(),position.y());
    if(points[layer].size()==0){
      minpos[layer][0] = position.x(); minpos[layer][1] = position.y();
      maxpos[layer][0] = position.x(); maxpos[layer][1] = position.y();
    }else{
      minpos[layer][0] = std::min((float)position.x(),minpos[layer][0]);
      minpos[layer][1] = std::min((float)position.y(),minpos[layer][1]);
      maxpos[layer][0] = std::max((float)position.x(),maxpos[layer][0]);
      maxpos[layer][1] = std::max((float)position.y(),maxpos[layer][1]);

    }
  }

}
// Create a vector of Hexels associated to one cluster from a collection of HGCalRecHits - this can be used
// directly to make the final cluster list - this method can be invoked multiple times for the same event
// with different input (reset should be called between events)
void HGCalImagingAlgo::makeClusters()
{
  //used for speedy search

  std::vector<KDTree> hit_kdtree(2*(maxlayer+1));

  //assign all hits in each layer to a cluster core or halo
  for (unsigned int i = 0; i <= 2*maxlayer+1; ++i) {
    KDTreeBox bounds(minpos[i][0],maxpos[i][0],
		     minpos[i][1],maxpos[i][1]);

    hit_kdtree[i].build(points[i],bounds);

    int actualLayer = int(abs(i-(maxlayer+1))); //maps back from index used for KD trees to actual layer

    double maxdensity = calculateLocalDensity(points[i],hit_kdtree[i], actualLayer);
    calculateDistanceToHigher(points[i],hit_kdtree[i]);
    findAndAssignClusters(points[i],hit_kdtree[i],maxdensity,bounds,actualLayer);
  }
  //make the cluster vector
}

std::vector<reco::BasicCluster> HGCalImagingAlgo::getClusters(bool doSharing){

  reco::CaloID caloID = reco::CaloID::DET_HGCAL_ENDCAP;
  std::vector< std::pair<DetId, float> > thisCluster;
  for (unsigned int i = 0; i < current_v.size(); ++i){
    double energy = 0;
    Point position;

    if( doSharing ) {

      std::vector<unsigned> seeds = findLocalMaximaInCluster(current_v[i]);
      // sharing found seeds.size() sub-cluster seeds in cluster i

      std::vector<std::vector<double> > fractions;
      // first pass can have noise it in
      shareEnergy(current_v[i],seeds,fractions);

      // reset and run second pass after vetoing seeds
      // that result in trivial clusters (less than 2 effective cells)


      for( unsigned isub = 0; isub < fractions.size(); ++isub ) {
	double effective_hits = 0.0;
	double energy  = calculateEnergyWithFraction(current_v[i],fractions[isub]);
	Point position = calculatePositionWithFraction(current_v[i],fractions[isub]);

	//std::cout << "Fractions*Energies: ";
	for( unsigned ihit = 0; ihit < fractions[isub].size(); ++ihit ) {
	  const double fraction = fractions[isub][ihit];
	  if( fraction > 1e-7 ) {
	    //std::cout << fraction << "*" << current_v[i][ihit].weight << " ";
	    effective_hits += fraction;
	    thisCluster.emplace_back(current_v[i][ihit].data.detid,fraction);
	  }
	}
	//std::cout << std::endl;

	if (verbosity < pINFO)
	  {
	    std::cout << "\t******** NEW CLUSTER (SHARING) ********" << std::endl;
	    std::cout << "\tEff. No. of cells = " << effective_hits << std::endl;
	    std::cout << "\t     Energy       = " << energy << std::endl;
	    std::cout << "\t     Phi          = " << position.phi() << std::endl;
	    std::cout << "\t     Eta          = " << position.eta() << std::endl;
	    std::cout << "\t*****************************" << std::endl;
	  }
	clusters_v.push_back(reco::BasicCluster(energy, position, caloID, thisCluster,
						algoId));
	thisCluster.clear();
      }
    }else{
      position = calculatePosition(current_v[i]);
      std::vector< KDNode >::iterator it;
      for (it = current_v[i].begin(); it != current_v[i].end(); it++)
	{
	  energy += (*it).data.isHalo ? 0. : (*it).data.weight;
	  thisCluster.emplace_back(std::pair<DetId, float>((*it).data.detid,((*it).data.isHalo?0.:1.)));
	};
      if (verbosity < pINFO)
	{
	  std::cout << "******** NEW CLUSTER (HGCIA) ********" << std::endl;
	  std::cout << "Index          " << i                   << std::endl;
	  std::cout << "No. of cells = " << current_v[i].size() << std::endl;
	  std::cout << "     Energy     = " << energy << std::endl;
	  std::cout << "     Phi        = " << position.phi() << std::endl;
	  std::cout << "     Eta        = " << position.eta() << std::endl;
	  std::cout << "*****************************" << std::endl;
	}
      clusters_v.push_back(reco::BasicCluster(energy, position, caloID, thisCluster,
					      algoId));
      thisCluster.clear();
    }
  }
  return clusters_v;
}

math::XYZPoint HGCalImagingAlgo::calculatePosition(std::vector<KDNode> &v){
  float total_weight = 0.;
  float x = 0.;
  float y = 0.;
  float z = 0.;
  for (unsigned int i = 0; i < v.size(); i++){
    if(!v[i].data.isHalo){
      total_weight += v[i].data.weight;
      x += v[i].data.x*v[i].data.weight;
      y += v[i].data.y*v[i].data.weight;
      z += v[i].data.z*v[i].data.weight;
    }
  }

  if (total_weight != 0) {
  return math::XYZPoint( x/total_weight,
			 y/total_weight,
			 z/total_weight );
  }
  return math::XYZPoint(0, 0, 0);
}

double HGCalImagingAlgo::calculateLocalDensity(std::vector<KDNode> &nd, KDTree &lp, const unsigned int layer){
  double maxdensity = 0.;
  float delta_c = 9999.;
  if( layer<= lastLayerEE ) delta_c = vecDeltas[0];
  else if( layer <= lastLayerFH) delta_c = vecDeltas[1];
  else delta_c = vecDeltas[2];
  for(unsigned int i = 0; i < nd.size(); ++i){
    KDTreeBox search_box(nd[i].dims[0]-delta_c,nd[i].dims[0]+delta_c,
			 nd[i].dims[1]-delta_c,nd[i].dims[1]+delta_c);
    std::vector<KDNode> found;
    lp.search(search_box,found);
    for(unsigned int j = 0; j < found.size(); j++){
      if(distance(nd[i].data,found[j].data) < delta_c){
	nd[i].data.rho += found[j].data.weight;
	if(nd[i].data.rho > maxdensity) maxdensity = nd[i].data.rho;
      }
    }
  }
  return maxdensity;
}

double HGCalImagingAlgo::calculateDistanceToHigher(std::vector<KDNode> &nd, KDTree &lp){


  //sort vector of Hexels by decreasing local density
  std::vector<size_t> rs = sorted_indices(nd);

  double maxdensity = 0.0;
  int nearestHigher = -1;


  if(rs.size()>0)
    maxdensity = nd[rs[0]].data.rho;
  else
    return maxdensity; // there are no hits
  double dist2 = 2500.0;
  //start by setting delta for the highest density hit to
  //the most distant hit - this is a convention

  for(unsigned int j = 0; j < nd.size(); j++){
    double tmp = distance2(nd[rs[0]].data, nd[j].data);
    dist2 = tmp > dist2 ? tmp : dist2;
  }
  nd[rs[0]].data.delta = std::sqrt(dist2);
  nd[rs[0]].data.nearestHigher = nearestHigher;

  //now we save the largest distance as a starting point

  const double max_dist2 = dist2;

  for(unsigned int oi = 1; oi < nd.size(); ++oi){ // start from second-highest density
    dist2 = max_dist2;
    unsigned int i = rs[oi];
    // we only need to check up to oi since hits
    // are ordered by decreasing density
    // and all points coming BEFORE oi are guaranteed to have higher rho
    // and the ones AFTER to have lower rho
    for(unsigned int oj = 0; oj < oi; oj++){
      unsigned int j = rs[oj];
      double tmp = distance2(nd[i].data, nd[j].data);
      if(tmp <= dist2){ //this "<=" instead of "<" addresses the (rare) case when there are only two hits
	dist2 = tmp;
	nearestHigher = j;
      }
    }
    nd[i].data.delta = std::sqrt(dist2);
    nd[i].data.nearestHigher = nearestHigher; //this uses the original unsorted hitlist
  }
  return maxdensity;
}

int HGCalImagingAlgo::findAndAssignClusters(std::vector<KDNode> &nd,KDTree &lp, double maxdensity, KDTreeBox &bounds, const int layer){

  //this is called once per layer...
  //so when filling the cluster temporary vector of Hexels we resize each time by the number
  //of clusters found. This is always equal to the number of cluster centers...

  unsigned int clusterIndex = 0;
  float delta_c = 9999.;
  if( layer<=28 ) delta_c = vecDeltas[0];
  else if( layer<=40 ) delta_c = vecDeltas[1];
  else delta_c = vecDeltas[2];

  std::vector<size_t> rs = sorted_indices(nd); // indices sorted by decreasing rho
  std::vector<size_t> ds = sort_by_delta(nd); // sort in decreasing distance to higher


  for(unsigned int i =0; i < nd.size(); ++i){

    if(nd[ds[i]].data.delta < delta_c) break; // no more cluster centers to be looked at
    if(dependSensor){

      float rho_c = kappa*nd[ds[i]].data.sigmaNoise;

      if(nd[ds[i]].data.rho < rho_c ) continue; // set equal to kappa times noise threshold
    }
    else if(nd[ds[i]].data.rho*kappa < maxdensity)
      continue;


    nd[ds[i]].data.clusterIndex = clusterIndex;
    if (verbosity < pINFO)
      {
	std::cout << "Adding new cluster with index " << clusterIndex+cluster_offset << std::endl;
	std::cout << "Cluster center is hit " << ds[i] << std::endl;
      }
    clusterIndex++;
  }

  //at this point clusterIndex is equal to the number of cluster centers - if it is zero we are
  //done
  if(clusterIndex==0) return clusterIndex;

  //assign to clusters, using the nearestHigher set from previous step (always set except
  // for top density hit that is skipped...
  for(unsigned int oi =1; oi < nd.size(); ++oi){
    unsigned int i = rs[oi];
    int ci = nd[i].data.clusterIndex;
    if(ci == -1){
      nd[i].data.clusterIndex =  nd[nd[i].data.nearestHigher].data.clusterIndex;
    }
  }

  //make room in the temporary cluster vector for the additional clusterIndex clusters
  // from this layer
  if (verbosity < pINFO)
    {
      std::cout << "resizing cluster vector by "<< clusterIndex << std::endl;
    }
  current_v.resize(cluster_offset+clusterIndex);

  //assign points closer than dc to other clusters to border region
  //and find critical border density
  std::vector<double> rho_b(clusterIndex,0.);
  lp.clear();
  lp.build(nd,bounds);
  //now loop on all hits again :( and check: if there are hits from another cluster within d_c -> flag as border hit
  for(unsigned int i = 0; i < nd.size(); ++i){
    int ci = nd[i].data.clusterIndex;
    bool flag_isolated = true;
    if(ci != -1){
      KDTreeBox search_box(nd[i].dims[0]-delta_c,nd[i].dims[0]+delta_c,
			   nd[i].dims[1]-delta_c,nd[i].dims[1]+delta_c);
      std::vector<KDNode> found;
      lp.search(search_box,found);

      for(unsigned int j = 1; j < found.size(); j++){
	//check if the hit is not within d_c of another cluster
	if(found[j].data.clusterIndex!=-1){
	  float dist = distance(found[j].data,nd[i].data);
	  if(dist < delta_c && found[j].data.clusterIndex!=ci){
	     //in which case we assign it to the border
	    nd[i].data.isBorder = true;
	    break;
	  }
	  //because we are using two different containers, we have to make sure that we don't unflag the
	  // hit when it finds *itself* closer than delta_c
	  if(dist < delta_c && dist != 0. && found[j].data.clusterIndex==ci){
	    //this is not an isolated hit
	    flag_isolated = false;
	  }
	}
      }
      if(flag_isolated) nd[i].data.isBorder = true; //the hit is more than delta_c from any of its brethren
    }
    //check if this border hit has density larger than the current rho_b and update
    if(nd[i].data.isBorder && rho_b[ci] < nd[i].data.rho)
      rho_b[ci] = nd[i].data.rho;
  }

  //flag points in cluster with density < rho_b as halo points, then fill the cluster vector
  for(unsigned int i = 0; i < nd.size(); ++i){
    int ci = nd[i].data.clusterIndex;
    if(ci!=-1 && nd[i].data.rho < rho_b[ci])
      nd[i].data.isHalo = true;
    if(nd[i].data.clusterIndex!=-1){
      current_v[ci+cluster_offset].push_back(nd[i]);
      if (verbosity < pINFO)
	{
	  std::cout << "Pushing hit " << i << " into cluster with index " << ci+cluster_offset << std::endl;
	  std::cout << "Size now " << current_v[ci+cluster_offset].size() << std::endl;
	}
    }
  }

  //prepare the offset for the next layer if there is one
  if (verbosity < pINFO)
    {
      std::cout << "moving cluster offset by " << clusterIndex << std::endl;
    }
  cluster_offset += clusterIndex;
  return clusterIndex;
}

// find local maxima within delta_c, marking the indices in the cluster
std::vector<unsigned> HGCalImagingAlgo::findLocalMaximaInCluster(const std::vector<KDNode>& cluster) {
  std::vector<unsigned> result;
  std::vector<bool> seed(cluster.size(),true);
  float delta_c = 2.;

  for( unsigned i = 0; i < cluster.size(); ++i ) {
    for( unsigned j = 0; j < cluster.size(); ++j ) {
      if( distance(cluster[i].data,cluster[j].data) < delta_c && i != j) {
	//std::cout << "hit-to-hit distance = " << distance(cluster[i],cluster[j]) << std::endl;
	if( cluster[i].data.weight < cluster[j].data.weight ) {
	  seed[i] = false;
	  break;
	}
      }
    }
  }

  for( unsigned i = 0 ; i < cluster.size(); ++i ) {
    if( seed[i] && cluster[i].data.weight > 5e-4) {
      // seed at i with energy cluster[i].weight
      result.push_back(i);
    }
  }

  // Found result.size() sub-clusters in input cluster of length cluster.size()

  return result;
}

math::XYZPoint HGCalImagingAlgo::calculatePositionWithFraction(const std::vector<KDNode>& hits,
								 const std::vector<double>& fractions) {
  double norm(0.0), x(0.0), y(0.0), z(0.0);
  for( unsigned i = 0; i < hits.size(); ++i ) {
    const double weight = fractions[i]*hits[i].data.weight;
    norm += weight;
    x += weight*hits[i].data.x;
    y += weight*hits[i].data.y;
    z += weight*hits[i].data.z;
  }
  math::XYZPoint result(x,y,z);
  double norm_inv = 1.0/norm;
  result *= norm_inv;
  return result;
}

double HGCalImagingAlgo::calculateEnergyWithFraction(const std::vector<KDNode>& hits,
						     const std::vector<double>& fractions) {
  double result = 0.0;
  for( unsigned i = 0 ; i < hits.size(); ++i ) {
    result += fractions[i]*hits[i].data.weight;
  }
  return result;
}

void HGCalImagingAlgo::shareEnergy(const std::vector<KDNode>& incluster,
				   const std::vector<unsigned>& seeds,
				   std::vector<std::vector<double> >& outclusters) {
  std::vector<bool> isaseed(incluster.size(),false);
  outclusters.clear();
  outclusters.resize(seeds.size());
  std::vector<Point> centroids(seeds.size());
  std::vector<double> energies(seeds.size());

  if( seeds.size() == 1 ) { // short circuit the case of a lone cluster
    outclusters[0].clear();
    outclusters[0].resize(incluster.size(),1.0);
    return;
  }

  //std::cout << "saving seeds" << std::endl;

  // create quick seed lookup
  for( unsigned i = 0; i < seeds.size(); ++i ) {
    isaseed[seeds[i]] = true;
  }

  // initialize clusters to be shared
  // centroids start off at seed positions
  // seeds always have fraction 1.0, to stabilize fit
  //std::cout << "initializing fit" << std::endl;
  for( unsigned i = 0; i < seeds.size(); ++i ) {
    outclusters[i].resize(incluster.size(),0.0);
    for( unsigned j = 0; j < incluster.size(); ++j ) {
      if( j == seeds[i] ) {
	outclusters[i][j] = 1.0;
	centroids[i] = math::XYZPoint(incluster[j].data.x,incluster[j].data.y,incluster[j].data.z);
	energies[i]  = incluster[j].data.weight;
      }
    }
  }

  // run the fit while we are less than max iterations, and clusters are still moving
  const double minFracTot = 1e-20;
  unsigned iter = 0;
  const unsigned iterMax = 50;
  double diff = std::numeric_limits<double>::max();
  const double stoppingTolerance = 1e-8;
  const double toleranceScaling = std::pow(std::max(1.0,seeds.size()-1.0),2.0);
  std::vector<Point> prevCentroids;
  std::vector<double> frac(seeds.size()), dist2(seeds.size());
  while( iter++ < iterMax && diff > stoppingTolerance*toleranceScaling ) {
    for( unsigned i = 0; i < incluster.size(); ++i ) {
      const Hexel& ihit = incluster[i].data;
      double fraction(0.0), fracTot(0.0), d2(0.0);
      for( unsigned j = 0; j < seeds.size(); ++j ) {
	fraction = 0.0;
	d2 = ( std::pow(ihit.x - centroids[j].x(),2.0) +
	       std::pow(ihit.y - centroids[j].y(),2.0) +
	       std::pow(ihit.z - centroids[j].z(),2.0)   )/sigma2;
	dist2[j] = d2;
	// now we set the fractions up based on hit type
	if( i == seeds[j] ) { // this cluster's seed
	  fraction = 1.0;
	} else if( isaseed[i] ) {
	  fraction = 0.0;
	} else {
	  fraction = energies[j]*std::exp( -0.5*d2 );
	}
	fracTot += fraction;
	frac[j] = fraction;
      }
      // now that we have calculated all fractions for all hits
      // assign the new fractions
      for( unsigned j = 0; j < seeds.size(); ++j ) {
	if( fracTot > minFracTot ||
	    ( i == seeds[j] && fracTot > 0.0 ) ) {
	  outclusters[j][i] = frac[j]/fracTot;
	} else {
	  outclusters[j][i] = 0.0;
	}
      }
    }

    // save previous centroids
    prevCentroids = std::move(centroids);
    // finally update the position of the centroids from the last iteration
    centroids.resize(seeds.size());
    double diff2 = 0.0;
    for( unsigned i = 0; i < seeds.size(); ++i ) {
      centroids[i] = calculatePositionWithFraction(incluster,outclusters[i]);
      energies[i]  = calculateEnergyWithFraction(incluster,outclusters[i]);
      // calculate convergence parameters
      const double delta2 = (prevCentroids[i]-centroids[i]).perp2();
      if( delta2 > diff2 ) diff2 = delta2;
    }
    //update convergance parameter outside loop
    diff = std::sqrt(diff2);
  }
}

void HGCalImagingAlgo::computeThreshold() {

  if(initialized) return;
  const std::vector<DetId>& listee(rhtools_.getGeometry()->getValidDetIds(DetId::Forward,ForwardSubdetector::HGCEE));
  const std::vector<DetId>& listfh(rhtools_.getGeometry()->getValidDetIds(DetId::Forward,ForwardSubdetector::HGCHEF));

  std::vector<double> dummy;
  dummy.resize(maxNumberOfWafersPerLayer, 0);
  thresholds.resize(maxlayer,dummy);
  float thickness=-9999;
  unsigned thickIndex = -1;
  float sigmaNoise = -9999.;
  int previouswafer=-999;
  int layer = -999;
  int wafer=-999;

  for(unsigned icalo=0;icalo<2;++icalo)
    {
      const std::vector<DetId>& listDetId( icalo==0 ? listee : listfh);
      unsigned nDetIds=listDetId.size();

      for(unsigned i=0;i<nDetIds;++i)
	{
	  HGCalDetId detid = listDetId[i];
	  wafer=detid.wafer();
	  if(wafer==previouswafer) continue;
	  previouswafer=detid.wafer();
	  // no need to do it twice
	  if(detid.zside()<0) continue;
	  layer = rhtools_.getLayerWithOffset(detid);

	  thickness = rhtools_.getSiThickness(detid);
	  if( thickness>99. && thickness<101.) thickIndex=0;
	  else if( thickness>199. && thickness<201. ) thickIndex=1;
	  else if( thickness>299. && thickness<301. ) thickIndex=2;
	  else assert( thickIndex>0 && "ERROR - silicon thickness has a nonsensical value" );
	  sigmaNoise = 0.001 * fcPerEle * nonAgedNoises[thickIndex] * dEdXweights[layer] / (fcPerMip[thickIndex] * thicknessCorrection[thickIndex]);
	  thresholds[layer-1][wafer]=sigmaNoise*ecut;
	}
    }

  // now BH, much faster
  for ( unsigned ilayer=layer+1;ilayer<=maxlayer;++ilayer)
    {
      sigmaNoise = 0.001 * noiseMip * dEdXweights[ilayer];
      dummy.clear();
      dummy.push_back(sigmaNoise*ecut);
      thresholds[ilayer-1]=dummy;
    }
  initialized=true;
}
