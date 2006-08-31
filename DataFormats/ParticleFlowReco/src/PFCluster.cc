#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace reco;

int    PFCluster::depthCorMode_ = 0;
double PFCluster::depthCorA_ = 0.89;
double PFCluster::depthCorB_ = 7.3;
double PFCluster::depthCorAp_ = 0.89;
double PFCluster::depthCorBp_ = 4.0;


PFCluster::PFCluster() :
  id_(0),
  type_(0),
  layer_(0),
  energy_(0),
  posCalcMode_(0),
  posCalcP1_(0),
  posCalcDepthCor_(false),
  color_(1)
{}


PFCluster::PFCluster(unsigned id, int type) : 
  id_(id), 
  type_(type), 
  layer_(0),
  energy_(0), 
  posCalcMode_(0),
  posCalcP1_(0),
  posCalcDepthCor_(false),
  color_(1)
{}
  

PFCluster::PFCluster(const PFCluster& other) :
  rechits_(other.rechits_),
  id_(other.id_),
  type_(other.type_),
  layer_(other.layer_), 
  energy_(other.energy_),
  posxyz_(other.posxyz_),
  posrep_(other.posrep_),
  posCalcMode_(other.posCalcMode_),
  posCalcP1_(other.posCalcP1_),
  posCalcDepthCor_(other.posCalcDepthCor_),
  color_(other.color_)
{}


void PFCluster::reset() {
  
  energy_ = 0;
  posxyz_ *= 0;
  posrep_ *= 0;
  
  rechits_.clear();
}


void PFCluster::addRecHit( const reco::PFRecHit& rechit, double fraction) {

  rechits_.push_back( reco::PFRecHitFraction(&rechit, fraction) );
}


void PFCluster::calculatePosition( int algo, double p1, bool depcor) {

  if( rechits_.empty() ) {
    cerr<<"PFCluster::calculatePosition: empty cluster!!!"<<endl; 
    return;
  }

  posCalcMode_ = algo;
  posCalcP1_ = p1;
  posCalcDepthCor_ = depcor;
  

  posxyz_.SetXYZ(0,0,0);

  energy_ = 0;
  layer_ = 0;

  double normalize = 0;

  // calculate total energy and average layer ---------- //

  double layer = 0;
  for (unsigned ic=0; ic<rechits_.size(); ic++ ) {

    const reco::PFRecHit* rechit = rechits_[ic].getRecHit();

    double frac =  rechits_[ic].getFraction();
    double theRecHitEnergy = rechit->energy() * frac;

    energy_ += theRecHitEnergy;
//     if ( rechit->layer() == LAYER_ECAL_BARREL ||
// 	 rechit->layer() == LAYER_ECAL_ENDCAP ) 
//       fEecal += theRecHitEnergy;
//     else if ( rechit->layer() == LAYER_HCAL_BARREL_1 ||
// 	      rechit->layer() == LAYER_HCAL_BARREL_2 ||
// 	      rechit->layer() == LAYER_HCAL_ENDCAP_1 ||
// 	      rechit->layer() == LAYER_HCAL_ENDCAP_2 || 
// 	      rechit->layer() == LAYER_VFCAL	      
// 	      ) 
//       fEhcal += theRecHitEnergy;

    layer += rechit->layer() * theRecHitEnergy;
  }  
  layer /= energy_;
  layer_ = lrintf(layer); // nearest integer
  
  if( p1 < 0 ) { 
    // automatic (and hopefully best !) determination of the parameter
    // for position determination.

    
    switch(algo) {
    case POSCALC_LIN: break;
    case POSCALC_LOG:
      switch(layer_) {
      case PFLayer::ECAL_BARREL:
	p1 = 0.004 + 0.022*energy_; // 27 feb 2006 
	break;
      case PFLayer::ECAL_ENDCAP:
	p1 = 0.014 + 0.009*energy_; // 27 feb 2006 
	break;
      case PFLayer::HCAL_BARREL1:
      case PFLayer::HCAL_BARREL2:
      case PFLayer::HCAL_ENDCAP:
      case PFLayer::VFCAL:
	// p1 = 0.007357 + 0.014952*energy_ + -0.000034*energy_*energy_; 
	p1 = 5.41215e-01 * log( energy_ / 1.29803e+01 );
	if(p1<0.01) p1 = 0.01;
	break;
      default:
	cerr<<"Clusters weight_p1_HCAL -1 NOT YET ALLOWED! Chose a better value in the opt file"<<endl;
	assert(0);
	break;
      }

      break;
    default:
      cerr<<"p1<0 means automatic p1 determination from the cluster energy"
	  <<endl;
      cerr<<"this is only implemented for POSCALC_LOG2, sorry."<<endl;
      cerr<<algo<<endl;
      assert(0);
      break;
    }
  } 

  // calculate uncorrected cluster position -------------------------

  REPPoint clusterpos;               // uncorrected cluster pos 
  math::XYZPoint clusterposxyz;      // idem, xyz coord 
  math::XYZPoint firstrechitposxyz;  // pos of the rechit with highest E

  double maxe = -9999;
  double x = 0;
  double y = 0;
  double z = 0;
  for (unsigned ic=0; ic<rechits_.size(); ic++ ) {
    
    const reco::PFRecHit* rechit = rechits_[ic].getRecHit();
    double fraction =  rechits_[ic].getFraction();  
    double theRecHitEnergy = rechit->energy() * fraction;

    double norm=0;
    
    switch(algo) {
    case POSCALC_LIN:
      norm = theRecHitEnergy;
      break;
    case POSCALC_LOG:
      norm =   max(0., log(theRecHitEnergy/p1 ));
      break;
    default:
      cerr<<"algo "<<algo<<" not defined !"<<endl;
      assert(0);
    }
    
    
    const math::XYZPoint& rechitposxyz = rechit->positionXYZ();
    
    if( theRecHitEnergy > maxe ) {
      firstrechitposxyz = rechitposxyz;
      maxe = theRecHitEnergy;
    }

    x += rechitposxyz.X() * norm;
    y += rechitposxyz.Y() * norm;
    z += rechitposxyz.Z() * norm;
    
    //     clusterposxyz += rechitposxyz * norm;
    normalize += norm;
  }
  
  // normalize uncorrected position
  // assert(normalize);
  if( normalize < 1e-9 ) {
    //    cerr<<"--------------------"<<endl;
    //    cerr<<(*this)<<endl;
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

  posrep_ = clusterpos;
  posxyz_ = clusterposxyz;


  // correction of the rechit position, 
  // according to the depth, only for ECAL 


  if( depthCorMode_ &&   // correction activated
      depcor &&          // correction requested
      ( layer_ == PFLayer::ECAL_BARREL ||       
	layer_ == PFLayer::ECAL_ENDCAP ) ) {

    posrep_.SetXYZ(0,0,0);
    normalize = 0;
    
    double corra = depthCorA_;
    double corrb = depthCorB_;
    if( abs(clusterpos.Eta() ) < 2.6 && 
	abs(clusterpos.Eta() ) > 1.65   ) { 
      // if crystals under preshower, correction is not the same  
      // (shower depth smaller)
      corra = depthCorAp_;
      corrb = depthCorBp_;
    }

    double depth = 0;
    switch( depthCorMode_ ) {
    case 1: // for e/gamma 
      depth = corra * ( corrb + log(energy_) ); 
      break;
    case 2: // for hadrons
      depth = corra;
      break;
    default:
      cerr<<"ClusterEF::calculatePosition : unknown function for depth correction! "<<endl;
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

    // cout<<"depth : "<<depth<<" "<<sqrt( depthv.Mag2() )<<endl;

    // now calculate corrected cluster position:    
    math::XYZPoint clusterposxyzcor;

    maxe = -9999;
    x = 0;
    y = 0;
    z = 0;
    for (unsigned ic=0; ic<rechits_.size(); ic++ ) {
      const reco::PFRecHit* rechit = rechits_[ic].getRecHit();
      double fraction =  rechits_[ic].getFraction();
      
      double theRecHitEnergy = rechit->energy() * fraction;

      const math::XYZPoint&  rechitposxyz = rechit->positionXYZ();
      const math::XYZVector& rechitaxis = rechit->getAxisXYZ();


//       // corrected rechit position:
//       math::XYZVector rechitposxyzcor(rechitaxis); 
//       rechitposxyzcor /= sqrt( rechitposxyzcor.Mag2() );    
//       rechitposxyzcor *= rechitaxis.Dot( depthv );

//       //      rechitposxyzcor.SetMag( rechitaxis.Dot( depthv ) );
//       rechitposxyzcor += rechitposxyz;

      math::XYZVector displacement( rechitaxis );
      displacement /= sqrt( displacement.Mag2() );    
      displacement *= displacement.Dot( depthv );
      
      math::XYZPoint rechitposxyzcor( rechitposxyz );
      rechitposxyzcor += displacement;

      // cout<<"displacement length "<<sqrt( displacement.Mag2() )<<endl;
      // cout<<"rechit pos "<<rechitposxyz.Rho()
      //	  <<" "<<rechitposxyz.Eta()
      //	  <<" "<<rechitposxyz.Phi()<<endl;
      // cout<<"rechit cor "<<rechitposxyzcor.Rho()
      //	  <<" "<<rechitposxyzcor.Eta()
      //	  <<" "<<rechitposxyzcor.Phi()<<endl;

      if( theRecHitEnergy > maxe ) {
	firstrechitposxyz = rechitposxyzcor;
	maxe = theRecHitEnergy;
      }
      
      double norm=0;
    
      switch(algo) {
      case POSCALC_LIN:
	norm = theRecHitEnergy;
	break;
      case POSCALC_LOG:
	norm = max(0., log(theRecHitEnergy/p1 ));
	break;
      default:
	cerr<<"algo "<<algo<<" not defined !"<<endl;
	assert(0);
      }
      
      x += rechitposxyzcor.X() * norm;
      y += rechitposxyzcor.Y() * norm;
      z += rechitposxyzcor.Z() * norm;
      
      // clusterposxyzcor += rechitposxyzcor * norm;
      normalize += norm;
    }
    
    // normalize
    if(normalize < 1e-9) {
      cerr<<"--------------------"<<endl;
      cerr<<(*this)<<endl;
      assert(0);
    }
    else {
      x /= normalize;
      y /= normalize;
      z /= normalize;
      

      clusterposxyzcor.SetCoordinates(x,y,z);
      posrep_.SetCoordinates( clusterposxyzcor.Rho(), 
			      clusterposxyzcor.Eta(), 
			      clusterposxyzcor.Phi() );
      posxyz_  = clusterposxyzcor;
      clusterposxyz = clusterposxyzcor;
    }
  }
  
//   // here calculate the distance between the cluster and the rechit with max e

//   fDseed = ( clusterposxyz - firstrechitposxyz).Mag();
  
//   // calculate the distance between the cluster and each rechit

//   for (unsigned ic=0; ic<rechits_.size(); ic++ ) {
//     rechits_[ic].SetDistToCluster( ( clusterposxyz - rechits_[ic].getRecHit()->positionXYZ() ).Mag() );
//   }

}


PFCluster& PFCluster::operator=(const PFCluster& other) {

  rechits_ = other.rechits_;
  id_ = other.id_;
  type_ = other.type_; 
  layer_= other.layer_;
  energy_ = other.energy_;
  posxyz_ = other.posxyz_;
  posrep_ = other.posrep_;
  posCalcMode_ = other.posCalcMode_;
  posCalcP1_ = other.posCalcP1_;
  posCalcDepthCor_ = other.posCalcDepthCor_;
  color_ = other.color_;

  return *this;
}


PFCluster& PFCluster::operator+=(const PFCluster& other) {
  
  if( id_==0 && 
      rechits_.empty() ) { // this is a default cluster
    (*this) = other;
    return *this;
  }

  // these clusters must be compatible
  assert( posCalcMode_ == other.posCalcMode_ &&
	  fabs(posCalcP1_ - other.posCalcP1_)<0.000001 && 
	  posCalcDepthCor_ == other.posCalcDepthCor_ && 
	  layer_ == other.layer_);

  for(unsigned ic=0; ic<other.rechits_.size(); ic++) {
    reco::PFRecHitFraction& cf 
      = const_cast<reco::PFRecHitFraction&>( other.rechits_[ic] );
    
    addRecHit( *(cf.getRecHit()) , cf.getFraction() );
  }

  // sortCells();
  calculatePosition(other.posCalcMode_, 
		    other.posCalcP1_, 
		    other.posCalcDepthCor_);
 
  cout<<"clusters have been added !"<<endl;
 
  return *this;
}

double PFCluster::getDepthCorrection(double energy, bool isBelowPS,
				     bool isHadron)
{
  double corrA = depthCorA_;
  double corrB = depthCorB_;
  if (isBelowPS) {
    corrA = depthCorAp_;
    corrB = depthCorBp_;
  }
  double depth = 0;
  switch(isHadron) {
  case 0: // e/gamma
    depth = corrA*(corrB + log(energy)); 
    break;
  case 1: // hadrons
    depth = corrA;
    break;
  default:
    edm::LogError("PFCluster") << "unknown function for depth correction!"
			       << std::endl;
  }
  return depth;
}

std::ostream& reco::operator<<(std::ostream& out, 
			       const PFCluster& cluster) {
  
  if(!out) return out;
  
  const PFCluster::REPPoint&  pos = cluster.positionREP();
  const std::vector< reco::PFRecHitFraction >& fracs = 
    cluster.getRecHitFractions();

  out<<"cluster "<<cluster.id()
     <<"\ttype: "<<cluster.type()
     <<"\tenergy: "<<cluster.energy()
     <<"\tposition: "
     <<pos.Rho()<<","<<pos.Eta()<<","<<pos.Phi()<<endl;
  out<<"\t"<<fracs.size()<<" rechits: "<<endl;
  for(unsigned i=0; i<fracs.size(); i++) {
    out<<fracs[i]<<endl;
  }

  return out;
}
