#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

using namespace reco;
using namespace std;

const unsigned    PFRecHit::nNeighbours_ = 8;
const unsigned    PFRecHit::nCorners_ = 4;

PFRecHit::PFRecHit() : 
  detId_(0),
  layer_(PFLayer::NONE),
  energy_(0.), 
  rescale_(1.),
  energyUp_(0.),
  // seedState_(-1), 
  position_(math::XYZPoint(0.,0.,0.)),
  posrep_(REPPoint(0.,0.,0.)) {

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( position_ );    
  }

  cornersrep_.reserve( nCorners_ );
  for ( unsigned i=0; i<nCorners_; ++i ) {
    cornersrep_.push_back( 
      REPPoint(cornersxyz_[i].Rho(),
	       cornersxyz_[i].Eta(),
	       cornersxyz_[i].Phi() ) );
  }

}


PFRecHit::PFRecHit(unsigned detId,
                   PFLayer::Layer layer, 
                   double energy, 
                   const math::XYZPoint& position,
                   const math::XYZVector& axisxyz,
                   const vector< math::XYZPoint >& cornersxyz) : 
  detId_(detId),
  layer_(layer),
  energy_(energy), 
  rescale_(1.),
  energyUp_(0.),
  // seedState_(-1), 
  position_(position),
  posrep_( position.Rho(), position.Eta(), position.Phi() ),
  axisxyz_(axisxyz),
  cornersxyz_(cornersxyz) 
{
  cornersrep_.reserve( nCorners_ );
  for ( unsigned i=0; i<nCorners_; ++i ) {
    cornersrep_.push_back( 
      REPPoint(cornersxyz_[i].Rho(),
	       cornersxyz_[i].Eta(),
	       cornersxyz_[i].Phi() ) );
  }
}

PFRecHit::PFRecHit(unsigned detId,
                   PFLayer::Layer layer,
                   double energy, 
                   double posx, double posy, double posz, 
                   double axisx, double axisy, double axisz) :

  detId_(detId),
  layer_(layer),
  energy_(energy), 
  rescale_(1.),
  energyUp_(0.),
  // seedState_(-1), 
  position_(posx, posy, posz),
  posrep_( position_.Rho(), 
           position_.Eta(), 
           position_.Phi() ),  
  axisxyz_(axisx, axisy, axisz) {
  

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( position_ );    
  } 

  cornersrep_.reserve( nCorners_ );
  for ( unsigned i=0; i<nCorners_; ++i ) {
    cornersrep_.push_back( 
      REPPoint(cornersxyz_[i].Rho(),
	       cornersxyz_[i].Eta(),
	       cornersxyz_[i].Phi() ) );
  }
 
}    


PFRecHit::PFRecHit(const PFRecHit& other) :
  detId_(other.detId_), 
  layer_(other.layer_), 
  energy_(other.energy_), 
  rescale_(other.rescale_), 
  energyUp_(other.energyUp_),
  // seedState_(other.seedState_),
  position_(other.position_), 
  posrep_(other.posrep_),
  axisxyz_(other.axisxyz_),
  cornersxyz_(other.cornersxyz_),
  cornersrep_(other.cornersrep_),
  //   neighbours_(other.neighbours_),
  neighbours4_(other.neighbours4_),
  neighbours8_(other.neighbours8_),
  neighboursIds4_(other.neighboursIds4_),
  neighboursIds8_(other.neighboursIds8_)
{}


PFRecHit::~PFRecHit() 
{}


const PFRecHit::REPPoint& 
PFRecHit::positionREP() const {
  //   if( posrep_ == REPPoint() ) {
  //     posrep_.SetCoordinates( position_.Rho(), position_.Eta(), position_.Phi() );
  //   }
  return posrep_;
}


void 
PFRecHit::calculatePositionREP() {
  posrep_.SetCoordinates( position_.Rho(),  position_.Eta(),  position_.Phi() );
  for ( unsigned i=0; i<nCorners_; ++i ) {
    cornersrep_[i].SetCoordinates(cornersxyz_[i].Rho(),cornersxyz_[i].Eta(),cornersxyz_[i].Phi() );
  }
}


// void PFRecHit::setNeighbours( const vector< unsigned >& neighbours ) {
//   if( neighbours.size() != nNeighbours_ ) 
//     throw cms::Exception("CellNeighbourVector") 
//       << "number of neighbours must be nNeighbours_";
  
// //   neighbours_.clear(); 
// //   neighboursIds4_.clear(); 
// //   neighboursIds8_.clear(); 
//   neighbours4_.clear(); 
//   neighbours8_.clear(); 
  
// //   neighboursIds4_.reserve( PFRecHit::nNeighbours_ ); 
// //   neighboursIds8_.reserve( PFRecHit::nNeighbours_ ); 

//   // space is reserved, but this vectors will not always have size 4 or 8.
//   // they contain the indices to the neighbours that are present in the rechit
//   // collection 
//   neighbours4_.reserve( PFRecHit::nNeighbours_ ); 
//   neighbours8_.reserve( PFRecHit::nNeighbours_ ); 
   
// //   neighboursIds_.reserve(nNeighbours_); 
// //   neighbours_.reserve(nNeighbours_); 

// //   neighbours_ = neighbours;

//   for(unsigned i=0; i<neighbours.size(); i++) {
//     if( neighbours[i] ) {
//       neighbours8_.push_back( neighbours[i] );      
// //       neighboursIds8_.push_back( neighbours[i]->detId() );  
//       if( !(i%2) ) {
//      neighbours4_.push_back( neighbours[i] );
// //   neighboursIds4_.push_back( neighbours[i]->detId() );  
//       }
//     }
//   }
// }


void PFRecHit::add4Neighbour(unsigned index) {
  neighbours4_.push_back( index );
  neighbours8_.push_back( index );
} 

void PFRecHit::add8Neighbour(unsigned index) {
  neighbours8_.push_back( index );
} 




// void PFRecHit::findPtrsToNeighbours( const std::map<unsigned,  
//                                   reco::PFRecHit* >& allhits ) {

//   neighbours4_.clear();
//   neighbours8_.clear();
  
//   typedef std::map<unsigned, reco::PFRecHit* >::const_iterator IDH;

//   for(unsigned inid = 0; inid<neighboursIds8_.size(); inid++) {
    
//     IDH ineighbour = allhits.find( neighboursIds8_[inid] );
//     if( ineighbour != allhits.end() ) {
//       neighbours8_.push_back( ineighbour->second );
//     }
//   }


//   for(unsigned inid = 0; inid<neighboursIds4_.size(); inid++) {
    
//     IDH ineighbour = allhits.find( neighboursIds4_[inid] );
//     if( ineighbour != allhits.end() ) {
//       neighbours4_.push_back( ineighbour->second );
//     }
//   }

// }


void PFRecHit::setNWCorner( double posx, double posy, double posz ) {
  setCorner(0, posx, posy, posz);
}


void PFRecHit::setSWCorner( double posx, double posy, double posz ) {
  setCorner(1, posx, posy, posz);
}


void PFRecHit::setSECorner( double posx, double posy, double posz ) {
  setCorner(2, posx, posy, posz);
}


void PFRecHit::setNECorner( double posx, double posy, double posz ) {
  setCorner(3, posx, posy, posz);
}


void PFRecHit::setCorner( unsigned i, double posx, double posy, double posz ) {
  assert( cornersxyz_.size() == nCorners_);
  assert( cornersrep_.size() == nCorners_);
  assert( i<cornersxyz_.size() );

  cornersxyz_[i] = math::XYZPoint( posx, posy, posz);
  cornersrep_[i] = REPPoint(cornersxyz_[i].Rho(),
			    cornersxyz_[i].Eta(),
			    cornersxyz_[i].Phi() );
}


bool PFRecHit::isNeighbour4(unsigned id) const {

  for(unsigned i=0; i<neighbours4_.size(); i++ )
    if( id == neighbours4_[i] ) return true;

  return false;           
}


bool PFRecHit::isNeighbour8(unsigned id) const {
  
  for(unsigned i=0; i<neighbours8_.size(); i++ )
    if( id == neighbours8_[i] ) return true;

  return false;           
}


void PFRecHit::size(double& deta, double& dphi) const {

  double minphi=9999;
  double maxphi=-9999;
  double mineta=9999;
  double maxeta=-9999;
  for ( unsigned ic=0; ic<cornersrep_.size(); ++ic ) { 
    double eta = cornersrep_[ic].Eta();
    double phi = cornersrep_[ic].Phi();
    
    if(phi>maxphi) maxphi=phi;
    if(phi<minphi) minphi=phi;
    if(eta>maxeta) maxeta=eta;
    if(eta<mineta) mineta=eta;    
  }

  deta = maxeta - mineta;
  dphi = maxphi - minphi;
}


ostream& reco::operator<<(ostream& out, const reco::PFRecHit& hit) {

  if(!out) return out;

  //   reco::PFRecHit& nshit = const_cast<reco::PFRecHit& >(hit);
  //   const reco::PFRecHit::REPPoint& posrep = nshit.positionREP();
  
  const  math::XYZPoint& posxyz = hit.position();

  out<<"hit id:"<<hit.detId()
     <<" l:"<<hit.layer()
     <<" E:"<<hit.energy()
     <<" t:"<<hit.time()
     <<" rep:"<<posxyz.Rho()<<","<<posxyz.Eta()<<","<<posxyz.Phi()<<"| N:";
  //     <<" SEED: "<<hit.seedState_;
  for(unsigned i=0; i<hit.neighbours8_.size(); i++ ) {
    out<<hit.neighbours8_[i]<<" ";
  }
  

  //   out<<endl;
  //   out<<"neighbours "<<endl;
  //   for(unsigned i=0; i<hit.neighbours8_.size(); i++ ) {
  //     out<<"\t"<< hit.neighbours8_[i]->detId()<<endl;
  //   }
  //   out<<"--"<<endl;
  //   for(unsigned i=0; i<hit.neighboursIds8_.size(); i++ ) {
  //     out<<"\t"<< (hit.neighboursIds8_[i])<<endl;
  //   }
  
  //   bool printcorners = false;

  //   if(printcorners) {
    
  //     out<<endl<<"corners : "<<endl;
  //     const std::vector< math::XYZPoint >& corners = hit.getCornersXYZ();
  //     for(unsigned i=0; i<corners.size(); i++) {
  //       out<<"\t"<<corners[i].X()<<","<<corners[i].Y()<<","<<corners[i].Z()<<endl;
  //     }
  //   }  

  return out;
}
