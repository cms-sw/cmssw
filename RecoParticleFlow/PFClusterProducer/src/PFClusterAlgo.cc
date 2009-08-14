#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "Math/GenVector/VectorUtil.h"


#include <stdexcept>
#include <string>

using namespace std;

unsigned PFClusterAlgo::prodNum_ = 1;

//for debug only 
//#define PFLOW_DEBUG

PFClusterAlgo::PFClusterAlgo() :
  pfClusters_( new vector<reco::PFCluster> ),
  threshBarrel_(0.),
  threshPtBarrel_(0.),
  threshSeedBarrel_(0.2),
  threshPtSeedBarrel_(0.),
  threshEndcap_(0.),
  threshPtEndcap_(0.),
  threshSeedEndcap_(0.6),
  threshPtSeedEndcap_(0.),
  nNeighbours_(4),
  posCalcNCrystal_(-1),
  posCalcP1_(-1),
  showerSigma_(5),
  useCornerCells_(false),
  debug_(false) {}



void PFClusterAlgo::doClustering( const PFRecHitHandle& rechitsHandle ) {
  rechitsHandle_ = rechitsHandle;
  doClustering( *rechitsHandle );
}

void PFClusterAlgo::doClustering( const reco::PFRecHitCollection& rechits ) {


  if(pfClusters_.get() ) pfClusters_->clear();
  else 
    pfClusters_.reset( new std::vector<reco::PFCluster> );


  eRecHits_.clear();

  bool initMask = false;
  if( mask_.size() != rechits.size() ) {
    initMask = true;
    mask_.clear();
    mask_.reserve( rechits.size() );

    if( ! mask_.empty() ) 
      cerr<<"PClusterAlgo::doClustering: map size should be "<<mask_.size()
	  <<". Will be reinitialized."<<endl;    
  }
  
  color_.clear(); 
  color_.reserve( rechits.size() );
  seedStates_.clear();
  seedStates_.reserve( rechits.size() );
  usedInTopo_.clear();
  usedInTopo_.reserve( rechits.size() );
  
  for ( unsigned i = 0; i < rechits.size(); i++ ) {
    eRecHits_.insert( make_pair( rechit(i, rechits).energy(), i) );
    if(initMask) mask_.push_back( true );
    color_.push_back( 0 );     
    seedStates_.push_back( UNKNOWN ); 
    usedInTopo_.push_back( false ); 
  }  

  // look for seeds.
  findSeeds( rechits );

  // build topological clusters around seeds
  buildTopoClusters( rechits );
  
  // look for PFClusters inside each topological cluster (one per seed)
  for(unsigned i=0; i<topoClusters_.size(); i++) {

    const std::vector< unsigned >& topocluster = topoClusters_[i];
    buildPFClusters( topocluster, rechits ); 
  }
}


void PFClusterAlgo::setMask( const std::vector<bool>& mask ) {
  mask_ = mask;
}




double PFClusterAlgo::parameter( Parameter paramtype, 
				 PFLayer::Layer layer)  const {
  

  double value = 0;

  switch( layer ) {
  case PFLayer::ECAL_BARREL:
  case PFLayer::HCAL_BARREL1:
  case PFLayer::HCAL_BARREL2: // I think this is HO. 
                              // should not do anything for HO !
    switch(paramtype) {
    case THRESH:
      value = threshBarrel_;
      break;
    case SEED_THRESH:
      value = threshSeedBarrel_;
      break;
    case PT_THRESH:
      value = threshPtBarrel_;
      break;
    case SEED_PT_THRESH:
      value = threshPtSeedBarrel_;
      break;
    default:
      cerr<<"PFClusterAlgo::parameter : unknown parameter type "
	  <<paramtype<<endl;
      assert(0);
    }
    break;
  case PFLayer::ECAL_ENDCAP:
  case PFLayer::HCAL_ENDCAP:
  case PFLayer::PS1:
  case PFLayer::PS2:
  case PFLayer::HF_EM:
  case PFLayer::HF_HAD:
    // and no particle flow in VFCAL
    switch(paramtype) {
    case THRESH:
      value = threshEndcap_;
      break;
    case SEED_THRESH:
      value = threshSeedEndcap_;
      break;
    case PT_THRESH:
      value = threshPtEndcap_;
      break;
    case SEED_PT_THRESH:
      value = threshPtSeedEndcap_;
      break;
    default:
      cerr<<"PFClusterAlgo::parameter : unknown parameter type "
	  <<paramtype<<endl;
      assert(0);
    }
    break;
  default:
    cerr<<"PFClusterAlgo::parameter : unknown layer "<<layer<<endl;
    assert(0);
    break;
  }

  return value;
}



void PFClusterAlgo::findSeeds( const reco::PFRecHitCollection& rechits ) {

  seeds_.clear();

  // should replace this by the message logger.
#ifdef PFLOW_DEBUG
  if(debug_) 
    cout<<"PFClusterAlgo::findSeeds : start"<<endl;
#endif


  // loop on rechits (sorted by decreasing energy - not E_T)
  for(EH ih = eRecHits_.begin(); ih != eRecHits_.end(); ih++ ) {

    unsigned  rhi      = ih->second; 

    if(! masked(rhi) ) continue;
    // rechit was asked to be processed

    double    rhenergy = ih->first;   
    const reco::PFRecHit& wannaBeSeed = rechit(rhi, rechits);
     
    if( seedStates_[rhi] == NO ) continue;
    // this hit was already tested, and is not a seed
 
    // determine seed energy threshold depending on the detector
    int layer = wannaBeSeed.layer();
    double seedThresh = parameter( SEED_THRESH, 
				   static_cast<PFLayer::Layer>(layer) );
    double seedPtThresh = parameter( SEED_PT_THRESH, 
				     static_cast<PFLayer::Layer>(layer) );


#ifdef PFLOW_DEBUG
    if(debug_) 
      cout<<"layer:"<<layer<<" seedThresh:"<<seedThresh<<endl;
#endif


    if( rhenergy < seedThresh || (seedPtThresh>0. && wannaBeSeed.pt2() < seedPtThresh*seedPtThresh )) {
      seedStates_[rhi] = NO; 
      continue;
    } 

      
    // Find the cell unused neighbours
    const vector<unsigned>* nbp;

    switch ( layer ) { 
    case PFLayer::ECAL_BARREL:         
    case PFLayer::ECAL_ENDCAP:       
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
    case PFLayer::HCAL_ENDCAP:
    case PFLayer::HF_EM:
    case PFLayer::HF_HAD:
      if( nNeighbours_ == 4 ) {
	nbp = & wannaBeSeed.neighbours4();
      }
      else if( nNeighbours_ == 8 )
	nbp = & wannaBeSeed.neighbours8();
      else {
	cerr<<"you're not allowed to set n neighbours to "
	    <<nNeighbours_<<endl;
	assert(0);
      }
      break;
    case PFLayer::PS1:       
    case PFLayer::PS2:     
      nbp = & wannaBeSeed.neighbours4();
      break;

    default:
      cerr<<"CellsEF::PhotonSeeds : unknown layer "<<layer<<endl;
      assert(0);
    }

    const vector<unsigned>& neighbours = *nbp;

      
    // Select as a seed if all neighbours have a smaller energy

    seedStates_[rhi] = YES;
    for(unsigned in=0; in<neighbours.size(); in++) {
	
      const reco::PFRecHit& neighbour = rechit( neighbours[in], 
						rechits ); 
	
      // one neighbour has a higher energy -> the tested rechit is not a seed
      if( neighbour.energy() > wannaBeSeed.energy() ) {
	seedStates_[rhi] = NO;
	break;
      }
    }
      
    if ( seedStates_[rhi] == YES ) {

      // seeds_ contains the indices of all seeds. 
      seeds_.push_back( rhi );
      
      // marking the rechit
      paint(rhi, SEED);
	
      // then all neighbours cannot be seeds and are flagged as such
      for(unsigned in=0; in<neighbours.size(); in++) {
	seedStates_[ neighbours[in] ] = NO;
      }
    }
  }  

#ifdef PFLOW_DEBUG
  if(debug_) 
    cout<<"PFClusterAlgo::findSeeds : done"<<endl;
#endif
}



  
void PFClusterAlgo::buildTopoClusters( const reco::PFRecHitCollection& rechits ){

  topoClusters_.clear(); 
  
#ifdef PFLOW_DEBUG
  if(debug_) 
    cout<<"PFClusterAlgo::buildTopoClusters start"<<endl;
#endif
  
  for(unsigned is = 0; is<seeds_.size(); is++) {
    
    unsigned rhi = seeds_[is];

    if( !masked(rhi) ) continue;
    // rechit was masked to be processed

    // already used in a topological cluster
    if( usedInTopo_[rhi] ) {
#ifdef PFLOW_DEBUG
      if(debug_) 
	cout<<rhi<<" used"<<endl; 
#endif
      continue;
    }
    
    vector< unsigned > topocluster;
    buildTopoCluster( topocluster, rhi, rechits );
   
    if(topocluster.empty() ) continue;
    
    topoClusters_.push_back( topocluster );
  } 

#ifdef PFLOW_DEBUG
  if(debug_) 
    cout<<"PFClusterAlgo::buildTopoClusters done"<<endl;
#endif
  
  return;
}


void 
PFClusterAlgo::buildTopoCluster( vector< unsigned >& cluster,
				 unsigned rhi, 
				 const reco::PFRecHitCollection& rechits ){


#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"PFClusterAlgo::buildTopoCluster in"<<endl;
#endif

  const reco::PFRecHit& rh = rechit( rhi, rechits); 

  double e = rh.energy();
  int layer = rh.layer();
  
  double thresh = parameter( THRESH, 
			     static_cast<PFLayer::Layer>(layer) );
  double ptThresh = parameter( PT_THRESH, 
			       static_cast<PFLayer::Layer>(layer) );


  if( e < thresh ||  ptThresh > 0. && rh.pt2() < ptThresh*ptThresh ) {
#ifdef PFLOW_DEBUG
    if(debug_)
      cout<<"return : "<<e<<"<"<<thresh<<endl; 
#endif
    return;
  }

  // add hit to cluster

  cluster.push_back( rhi );
  // idUsedRecHits_.insert( rh.detId() );

  usedInTopo_[ rhi ] = true;

  //   cout<<" hit ptr "<<hit<<endl;

  // get neighbours
  const std::vector< unsigned >& nbs4 = rh.neighbours4();
  const std::vector< unsigned >& nbs8 = rh.neighbours8();
  // topo-cluster is computed from cells with 1 common side
  std::vector< unsigned > nbs = nbs4;
  // or cells with 1 common corner
  if( useCornerCells_ ) 
    nbs = nbs8;
  
  for(unsigned i=0; i<nbs.size(); i++) {

//     const reco::PFRecHit& neighbour = rechit( nbs[i], rechits );

//     set<unsigned>::iterator used 
//       = idUsedRecHits_.find( neighbour.detId() );
//     if(used != idUsedRecHits_.end() ) continue;
    
    // already used
    if( usedInTopo_[ nbs[i] ] ) {
#ifdef PFLOW_DEBUG
      if(debug_) 
	cout<<rhi<<" used"<<endl; 
#endif
      continue;
    }
			     
    buildTopoCluster( cluster, nbs[i], rechits );
  }
#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"PFClusterAlgo::buildTopoCluster out"<<endl;
#endif

}


void 
PFClusterAlgo::buildPFClusters( const std::vector< unsigned >& topocluster,
				const reco::PFRecHitCollection& rechits ) 
{


  //  bool debug = false;


  // several rechits may be seeds. initialize PFClusters on these seeds. 
  
  vector<reco::PFCluster> curpfclusters;
  vector< unsigned > seedsintopocluster;


  for(unsigned i=0; i<topocluster.size(); i++ ) {

    unsigned rhi = topocluster[i];

    if( seedStates_[rhi] == YES ) {

      reco::PFCluster cluster;

      double fraction = 1.0; 
      
      reco::PFRecHitRef  recHitRef = createRecHitRef( rechits, rhi ); 
	
      cluster.addRecHitFraction( reco::PFRecHitFraction( recHitRef, 
							 fraction ) );

    // cluster.addRecHit( rhi, fraction );
      
      calculateClusterPosition( cluster, 
			        true );    


//       cout<<"PFClusterAlgo: 2"<<endl;
      curpfclusters.push_back( cluster );
#ifdef PFLOW_DEBUG
      if(debug_) {
	cout << "PFClusterAlgo::buildPFClusters: seed "
	     << rechit( rhi, rechits) <<endl;
	cout << "PFClusterAlgo::buildPFClusters: pfcluster initialized : "
	     << cluster <<endl;
      }
#endif

      // keep track of the seed of each topocluster
      seedsintopocluster.push_back( rhi );
      
    }
  }

  // if only one seed in the topocluster, use all crystals
  // in the position calculation (posCalcNCrystal = -1)
  // otherwise, use the user specified value
  int posCalcNCrystal = seedsintopocluster.size()>1 ? posCalcNCrystal_:-1;
    
  // Find iteratively the energy and position
  // of each pfcluster in the topological cluster
  unsigned iter = 0;
  unsigned niter = 50;
  double diff = 1.;

  // if(debug_) niter=2;
  while ( iter++ < niter && diff > 1E-8 ) {

    // Store previous iteration's result and reset pfclusters     
    vector<double> ener;
    vector<math::XYZVector> tmp;

    for ( unsigned ic=0; ic<curpfclusters.size(); ic++ ) {
      ener.push_back( curpfclusters[ic].energy() );
      
      math::XYZVector v;
      v = curpfclusters[ic].position();

      tmp.push_back( v );

#ifdef PFLOW_DEBUG
      if(debug_)  {
	cout<<"saving photon pos "<<ic<<" "<<curpfclusters[ic]<<endl;
	cout<<tmp[ic].X()<<" "<<tmp[ic].Y()<<" "<<tmp[ic].Z()<<endl;
      }
#endif

      curpfclusters[ic].reset();
    }


    // Loop over topocluster cells
    for( unsigned irh=0; irh<topocluster.size(); irh++ ) {
      
      unsigned rhindex = topocluster[irh];
      
      const reco::PFRecHit& rh = rechit( rhindex, rechits);
      
      // int layer = rh.layer();
             
      vector<double> dist;
      vector<double> frac;
      double fractot = 0.;

      bool isaseed = isSeed(rhindex);

      math::XYZVector cposxyzcell;
      cposxyzcell = rh.position();

#ifdef PFLOW_DEBUG
      if(debug_) { 
	cout<<rh<<endl;
	cout<<"start loop on curpfclusters"<<endl;
      }
#endif

      // Loop over pfclusters
      for ( unsigned ic=0; ic<tmp.size(); ic++) {
	
#ifdef PFLOW_DEBUG
	if(debug_) cout<<"pfcluster "<<ic<<endl;
#endif
	
	double frc=0.;
	bool seedexclusion=true;

	// convert cluster coordinates to xyz
	math::XYZVector cposxyzclust( tmp[ic].X(), tmp[ic].Y(), tmp[ic].Z() );
	
#ifdef PFLOW_DEBUG
	if(debug_) {
	  
	  cout<<"CLUSTER "<<cposxyzclust.X()<<","
	      <<cposxyzclust.Y()<<","
	      <<cposxyzclust.Z()<<"\t\t"
	      <<"CELL "<<cposxyzcell.X()<<","
	      <<cposxyzcell.Y()<<","
	      <<cposxyzcell.Z()<<endl;
	}  
#endif
	
	// Compute the distance between the current cell 
	// and the current PF cluster, normalized to a 
	// number of "sigma"
	math::XYZVector deltav = cposxyzclust;
	deltav -= cposxyzcell;
	double d = deltav.R() / showerSigma_;
	
	// if distance cell-cluster is too large, it means that 
	// we're right on the junction between 2 subdetectors (HCAL/VFCAL...)
	// in this case, distance is calculated in the xy plane
	// could also be a large supercluster... 
#ifdef PFLOW_DEBUG
	if( d > 6. && debug_ ) { 
	  for( unsigned jrh=0; jrh<topocluster.size(); jrh++ ) {
	    paint(jrh, SPECIAL);
	  }
	  cout<<"PFClusterAlgo Warning: distance too large"<<d<<endl;
	}
#endif
	dist.push_back( d );

	// the current cell is the seed from the current photon.
	if( rhindex == seedsintopocluster[ic] && seedexclusion ) {
	  frc = 1.;
#ifdef PFLOW_DEBUG
	  if(debug_) cout<<"this cell is a seed for the current photon"<<endl;
#endif
	}
	else if( isaseed && seedexclusion ) {
	  frc = 0.;
#ifdef PFLOW_DEBUG
	  if(debug_) cout<<"this cell is a seed for another photon"<<endl;
#endif
	}
	else {
	  // Compute the fractions of the cell energy to be assigned to 
	  // each curpfclusters in the cluster.
	  frc = ener[ic] * exp ( - dist[ic]*dist[ic] / 2. );

#ifdef PFLOW_DEBUG
	  if(debug_) {
	    cout<<"dist["<<ic<<"] "<<dist[ic]
		<<", sigma="<<sigma
		<<", frc="<<frc<<endl;
	  }  
#endif
	
	}
	fractot += frc;
	frac.push_back(frc);
      }      

      // Add the relevant fraction of the cell to the curpfclusters
#ifdef PFLOW_DEBUG
      if(debug_) cout<<"start add cell"<<endl;
#endif
      for ( unsigned ic=0; ic<tmp.size(); ++ic ) {
#ifdef PFLOW_DEBUG
	if(debug_) 
	  cout<<" frac["<<ic<<"] "<<frac[ic]<<" "<<fractot<<" "<<rh<<endl;
#endif

	if( fractot ) 
	  frac[ic] /= fractot;
	else { 
#ifdef PFLOW_DEBUG
	  if( debug_ ) {
	    int layer = rh.layer();
	    cerr<<"fractot = 0 ! "<<layer<<endl;
	    
	    for( unsigned trh=0; trh<topocluster.size(); trh++ ) {
	      unsigned tindex = topocluster[trh];
	      const reco::PFRecHit& rh = rechit( tindex, rechits);
	      cout<<rh<<endl;
	    }

	    // assert(0)
	  }
#endif

	  continue;
	}

	// if the fraction has been set to 0, the cell 
	// is now added to the cluster - careful ! (PJ, 19/07/08)
	// BUT KEEP ONLY CLOSE CELLS OTHERWISE MEMORY JUST EXPLOSES
	// (PJ, 15/09/08 <- similar to what existed before the 
        // previous bug fix, but keeps the close seeds inside, 
	// even if their fraction was set to zero.)
	// Also add a protection to keep the seed in the cluster 
	// when the latter gets far from the former. These cases
	// (about 1% of the clusters) need to be studied, as 
	// they create fake photons, in general.
	// (PJ, 16/09/08) 
      	if ( dist[ic] < 6. || frac[ic] > 0.99999 ) { 
	  // if ( dist[ic] > 6. ) cout << "Warning : PCluster is getting very far from its seeding cell" << endl;
	  reco::PFRecHitRef  recHitRef = createRecHitRef( rechits, rhindex ); 
	  reco::PFRecHitFraction rhf( recHitRef,frac[ic] );
	  curpfclusters[ic].addRecHitFraction( rhf );
	}
      }
      // if(debug_) cout<<" end add cell"<<endl;
      
      dist.clear();
      frac.clear();
    }
    
    // Determine the new cluster position and check 
    // the distance with the previous iteration
    diff = 0.;
    for (  unsigned ic=0; ic<tmp.size(); ++ic ) {
      calculateClusterPosition(curpfclusters[ic], true, posCalcNCrystal);
#ifdef PFLOW_DEBUG
      if(debug_) cout<<"new iter "<<ic<<endl;
      if(debug_) cout<<curpfclusters[ic]<<endl;
#endif

      double delta = ROOT::Math::VectorUtil::DeltaR(curpfclusters[ic].position(),tmp[ic]);
      if ( delta > diff ) diff = delta;
    }
    ener.clear();
    tmp.clear();
  }
  
  // Issue a warning message if the number of iterations 
  // exceeds 50
#ifdef PFLOW_DEBUG
  if ( iter >= 50 && debug_ ) 
    cout << "PFClusterAlgo Warning: "
	 << "more than "<<niter<<" iterations in pfcluster finding: " 
	 <<  setprecision(10) << diff << endl;
#endif
  
  // There we go
  // add all clusters to the list of pfClusters.
  for(unsigned ic=0; ic<curpfclusters.size(); ic++) {
    calculateClusterPosition(curpfclusters[ic], true, posCalcNCrystal);
    pfClusters_->push_back(curpfclusters[ic]); 
  }
}



void 
PFClusterAlgo::calculateClusterPosition(reco::PFCluster& cluster,
					bool depcor, 
					int posCalcNCrystal) {

  if( posCalcNCrystal_ != -1 && 
      posCalcNCrystal_ != 5 && 
      posCalcNCrystal_ != 9 ) {
    throw "PFCluster::calculatePosition : posCalcNCrystal_ must be -1, 5, or 9.";
  }  

  if(!posCalcNCrystal) posCalcNCrystal = posCalcNCrystal_; 

  cluster.position_.SetXYZ(0,0,0);

  cluster.energy_ = 0;
  
  double normalize = 0;

  // calculate total energy, average layer, and look for seed  ---------- //

  // double layer = 0;
  map <PFLayer::Layer, double> layers; 
  unsigned seedIndex = 0;
  bool     seedIndexFound = false;

  //Colin: the following can be simplified!

  // loop on rechit fractions
  for (unsigned ic=0; ic<cluster.rechits_.size(); ic++ ) {

    unsigned rhi = cluster.rechits_[ic].recHitRef().index();
    // const reco::PFRecHit& rh = rechit( rhi, rechits );

    const reco::PFRecHit& rh = *(cluster.rechits_[ic].recHitRef());
    double fraction =  cluster.rechits_[ic].fraction();

    // Find the seed of this sub-cluster (excluding other seeds found in the topological
    // cluster, the energy fraction of which were set to 0 fpr the position determination.
    if( isSeed(rhi) && fraction > 1e-9 ) {
      seedIndex = rhi;
      seedIndexFound = true;
    }


    double recHitEnergy = rh.energy() * fraction;
    cluster.energy_ += recHitEnergy;

    // sum energy in each layer
    PFLayer::Layer layer = rh.layer();                              
    map <PFLayer::Layer, double>:: iterator it = layers.find(layer);
    if (it != layers.end()) 
      it->second += recHitEnergy;
    else 
      layers.insert(make_pair(layer, recHitEnergy));
  }  

  assert(seedIndexFound);

  // loop over pairs to find layer with max energy          
  double Emax = 0.;
  PFLayer::Layer layer = PFLayer::NONE;
  for (map<PFLayer::Layer, double>::iterator it = layers.begin();
       it != layers.end(); ++it) {
    double e = it->second;
    if(e > Emax){ 
      Emax = e; 
      layer = it->first;
    }
  }
  
  //setlayer here
  cluster.setLayer( layer ); // take layer with max energy

  // layer /= cluster.energy_;
  // cluster.layer_ = lrintf(layer); // nearest integer

  double p1 =  posCalcP1_;
  if( p1 < 0 ) { 
    // automatic (and hopefully best !) determination of the parameter
    // for position determination.
    
    // Remove the ad-hoc determination of p1, and set it to the 
    // seed threshold.
    switch(cluster.layer() ) {
    case PFLayer::ECAL_BARREL:
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
      p1 = threshBarrel_;
      break;
    case PFLayer::ECAL_ENDCAP:
    case PFLayer::HCAL_ENDCAP:
    case PFLayer::HF_EM:
    case PFLayer::HF_HAD:
    case PFLayer::PS1:
    case PFLayer::PS2:
      p1 = threshEndcap_;
      break;

    /*
    switch(cluster.layer() ) {
    case PFLayer::ECAL_BARREL:
      p1 = 0.004 + 0.022*cluster.energy_; // 27 feb 2006 
      break;
    case PFLayer::ECAL_ENDCAP:
      p1 = 0.014 + 0.009*cluster.energy_; // 27 feb 2006 
      break;
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
    case PFLayer::HCAL_ENDCAP:
    case PFLayer::HCAL_HF:
      p1 = 5.41215e-01 * log( cluster.energy_ / 1.29803e+01 );
      if(p1<0.01) p1 = 0.01;
      break;
    */

    default:
      cerr<<"Clusters weight_p1 -1 not yet allowed for layer "<<layer
	  <<". Chose a better value in the opt file"<<endl;
      assert(0);
      break;
    }
  } 
  else if( p1< 1e-9 ) { // will divide by p1 later on
    p1 = 1e-9;
  }

  // calculate uncorrected cluster position --------------------------------

  reco::PFCluster::REPPoint clusterpos;   // uncorrected cluster pos 
  math::XYZPoint clusterposxyz;           // idem, xyz coord 
  math::XYZPoint firstrechitposxyz;       // pos of the rechit with highest E

  double maxe = -9999;
  double x = 0;
  double y = 0;
  double z = 0;
  
  for (unsigned ic=0; ic<cluster.rechits_.size(); ic++ ) {
    
    unsigned rhi = cluster.rechits_[ic].recHitRef().index();
//     const reco::PFRecHit& rh = rechit( rhi, rechits );

    const reco::PFRecHit& rh = *(cluster.rechits_[ic].recHitRef());

    if(rhi != seedIndex) { // not the seed
      if( posCalcNCrystal == 5 ) { // pos calculated from the 5 neighbours only
	if(!rh.isNeighbour4(seedIndex) ) {
	  continue;
	}
      }
      if( posCalcNCrystal == 9 ) { // pos calculated from the 9 neighbours only
	if(!rh.isNeighbour8(seedIndex) ) {
	  continue;
	}
      }
    }
    
    double fraction =  cluster.rechits_[ic].fraction();
    double recHitEnergy = rh.energy() * fraction;

    double norm = fraction < 1E-9 ? 0. : max(0., log(recHitEnergy/p1 ));
    
    const math::XYZPoint& rechitposxyz = rh.position();
    
    if( recHitEnergy > maxe ) {
      firstrechitposxyz = rechitposxyz;
      maxe = recHitEnergy;
    }

    x += rechitposxyz.X() * norm;
    y += rechitposxyz.Y() * norm;
    z += rechitposxyz.Z() * norm;
    
    // clusterposxyz += rechitposxyz * norm;
    normalize += norm;
  }
  
  // normalize uncorrected position
  // assert(normalize);
  if( normalize < 1e-9 ) {
    //    cerr<<"--------------------"<<endl;
    //    cerr<<(*this)<<endl;
    cout << "Watch out : cluster too far from its seeding cell, set to 0,0,0" << endl;
    clusterposxyz.SetXYZ(0,0,0);
    clusterpos.SetCoordinates(0,0,0);
    return;
  }
  else {
    x /= normalize;
    y /= normalize; 
    z /= normalize; 
    clusterposxyz.SetCoordinates( x, y, z);
    clusterpos.SetCoordinates( clusterposxyz.Rho(), clusterposxyz.Eta(), clusterposxyz.Phi() );
  }  

  cluster.posrep_ = clusterpos;
  cluster.position_ = clusterposxyz;


  // correction of the rechit position, 
  // according to the depth, only for ECAL 


  if( depcor &&   // correction requested and ECAL
      ( cluster.layer() == PFLayer::ECAL_BARREL ||       
	cluster.layer() == PFLayer::ECAL_ENDCAP ) ) {

    
    double corra = reco::PFCluster::depthCorA_;
    double corrb = reco::PFCluster::depthCorB_;
    if( abs(clusterpos.Eta() ) < 2.6 && 
	abs(clusterpos.Eta() ) > 1.65   ) { 
      // if crystals under preshower, correction is not the same  
      // (shower depth smaller)
      corra = reco::PFCluster::depthCorAp_;
      corrb = reco::PFCluster::depthCorBp_;
    }

    double depth = 0;

    switch( reco::PFCluster::depthCorMode_ ) {
    case 1: // for e/gamma 
      depth = corra * ( corrb + log(cluster.energy_) ); 
      break;
    case 2: // for hadrons
      depth = corra;
      break;
    default:
      cerr<<"PFClusterAlgo::calculateClusterPosition : unknown function for depth correction! "<<endl;
      assert(0);
    }

    // calculate depth vector:
    // its mag is depth
    // its direction is the cluster direction (uncorrected)

//     double xdepthv = clusterposxyz.X();
//     double ydepthv = clusterposxyz.Y();
//     double zdepthv = clusterposxyz.Z();
//     double mag = sqrt( xdepthv*xdepthv + 
// 		       ydepthv*ydepthv + 
// 		       zdepthv*zdepthv );
    

//     math::XYZPoint depthv(clusterposxyz); 
//     depthv.SetMag(depth);
    
    
    math::XYZVector depthv( clusterposxyz.X(), 
			    clusterposxyz.Y(),
			    clusterposxyz.Z() );
    depthv /= sqrt(depthv.Mag2() );
    depthv *= depth;


    // now calculate corrected cluster position:    
    math::XYZPoint clusterposxyzcor;

    maxe = -9999;
    x = 0;
    y = 0;
    z = 0;
    cluster.posrep_.SetXYZ(0,0,0);
    normalize = 0;
    for (unsigned ic=0; ic<cluster.rechits_.size(); ic++ ) {

      unsigned rhi = cluster.rechits_[ic].recHitRef().index();
//       const reco::PFRecHit& rh = rechit( rhi, rechits );
      
      const reco::PFRecHit& rh = *(cluster.rechits_[ic].recHitRef());

      if(rhi != seedIndex) {
	if( posCalcNCrystal == 5 ) {
	  if(!rh.isNeighbour4(seedIndex) ) {
	    continue;
	  }
	}
	if( posCalcNCrystal == 9 ) {
	  if(!rh.isNeighbour8(seedIndex) ) {
	    continue;
	  }
	}
      }
    
      double fraction =  cluster.rechits_[ic].fraction();
      double recHitEnergy = rh.energy() * fraction;
      
      const math::XYZPoint&  rechitposxyz = rh.position();

      // rechit axis not correct ! 
      math::XYZVector rechitaxis = rh.getAxisXYZ();
      // rechitaxis -= math::XYZVector( rechitposxyz.X(), rechitposxyz.Y(), rechitposxyz.Z() );
      
      math::XYZVector rechitaxisu( rechitaxis );
      rechitaxisu /= sqrt( rechitaxis.Mag2() );

      math::XYZVector displacement( rechitaxisu );
      // displacement /= sqrt( displacement.Mag2() );    
      displacement *= rechitaxisu.Dot( depthv );
      
      math::XYZPoint rechitposxyzcor( rechitposxyz );
      rechitposxyzcor += displacement;

      if( recHitEnergy > maxe ) {
	firstrechitposxyz = rechitposxyzcor;
	maxe = recHitEnergy;
      }

      double norm = fraction < 1E-9 ? 0. : max(0., log(recHitEnergy/p1 ));
      
      x += rechitposxyzcor.X() * norm;
      y += rechitposxyzcor.Y() * norm;
      z += rechitposxyzcor.Z() * norm;
      
      // clusterposxyzcor += rechitposxyzcor * norm;
      normalize += norm;
    }
    
    // normalize
    if(normalize < 1e-9) {
      cerr<<"--------------------"<<endl;
      cerr<< cluster <<endl;
      assert(0);
    }
    else {
      x /= normalize;
      y /= normalize;
      z /= normalize;
      

      clusterposxyzcor.SetCoordinates(x,y,z);
      cluster.posrep_.SetCoordinates( clusterposxyzcor.Rho(), 
				      clusterposxyzcor.Eta(), 
				      clusterposxyzcor.Phi() );
      cluster.position_  = clusterposxyzcor;
      clusterposxyz = clusterposxyzcor;
    }
  }
}



const reco::PFRecHit& 
PFClusterAlgo::rechit(unsigned i, 
		      const reco::PFRecHitCollection& rechits ) {

  if( i < 0 || i >= rechits.size() ) {
    string err = "PFClusterAlgo::rechit : out of range";
    throw std::out_of_range(err);
  }
  
  return rechits[i];
}



bool PFClusterAlgo::masked(unsigned rhi) const {

  if(rhi<0 || rhi>=mask_.size() ) {
    string err = "PFClusterAlgo::masked : out of range";
    throw std::out_of_range(err);
  }
  
  return mask_[rhi];
}


unsigned PFClusterAlgo::color(unsigned rhi) const {

  if(rhi<0 || rhi>=color_.size() ) {
    string err = "PFClusterAlgo::color : out of range";
    throw std::out_of_range(err);
  }
  
  return color_[rhi];
}



bool PFClusterAlgo::isSeed(unsigned rhi) const {

  if(rhi<0 || rhi>=seedStates_.size() ) {
    string err = "PFClusterAlgo::isSeed : out of range";
    throw std::out_of_range(err);
  }
  
  return seedStates_[rhi]>0 ? true : false;
}


void PFClusterAlgo::paint(unsigned rhi, unsigned color ) {

  if(rhi<0 || rhi>=color_.size() ) {
    string err = "PFClusterAlgo::color : out of range";
    throw std::out_of_range(err);
  }
  
  color_[rhi] = color;
}


reco::PFRecHitRef 
PFClusterAlgo::createRecHitRef( const reco::PFRecHitCollection& rechits, 
				unsigned rhi ) {

  if( rechitsHandle_.isValid() ) {
    return reco::PFRecHitRef(  rechitsHandle_, rhi );
  } 
  else {
    return reco::PFRecHitRef(  &rechits, rhi );
  }
}


ostream& operator<<(ostream& out,const PFClusterAlgo& algo) {
  if(!out) return out;
  out<<"PFClusterAlgo parameters : "<<endl;
  out<<"-----------------------------------------------------"<<endl;
  out<<"threshBarrel     : "<<algo.threshBarrel_     <<endl;
  out<<"threshSeedBarrel : "<<algo.threshSeedBarrel_ <<endl;
  out<<"threshEndcap     : "<<algo.threshEndcap_     <<endl;
  out<<"threshSeedEndcap : "<<algo.threshSeedEndcap_ <<endl;
  out<<"nNeighbours      : "<<algo.nNeighbours_      <<endl;
  out<<"posCalcNCrystal  : "<<algo.posCalcNCrystal_  <<endl;
  out<<"posCalcP1        : "<<algo.posCalcP1_        <<endl;
  out<<"showerSigma      : "<<algo.showerSigma_      <<endl;
  out<<"useCornerCells   : "<<algo.useCornerCells_   <<endl;

  out<<endl;
  out<<algo.pfClusters_->size()<<" clusters:"<<endl;

  for(unsigned i=0; i<algo.pfClusters_->size(); i++) {
    out<<(*algo.pfClusters_)[i]<<endl;
    
    if(!out) return out;
  }
  
  return out;
}
