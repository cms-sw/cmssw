#include "DataFormats/PFReco/interface/PFRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;
using namespace std;

const unsigned    PFRecHit::nNeighbours_ = 8;
const unsigned    PFRecHit::nCorners_ = 4;

PFRecHit::PFRecHit() : 
  detId_(0),
  layer_(0),
  energy_(0.), 
  isSeed_(-1), 
  posxyz_(math::XYZPoint(0.,0.,0.)),
  posrep_(REPPoint(0.,0.,0.)) {

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( posxyz_ );    
  }
}


PFRecHit::PFRecHit(unsigned detId,
		   int layer, 
		   double energy, 
		   const math::XYZPoint& position,
		   const math::XYZVector& axisxyz,
		   const vector< math::XYZPoint >& cornersxyz) : 
  detId_(detId),
  layer_(layer),
  energy_(energy), 
  isSeed_(-1), 
  posxyz_(position),
  posrep_(REPPoint(0.,0.,0.)),
  axisxyz_(axisxyz),
  cornersxyz_(cornersxyz) {
}

PFRecHit::PFRecHit(unsigned detId,
		   int layer,
		   double energy, 
		   double posx, double posy, double posz, 
		   double axisx, double axisy, double axisz) :

  detId_(detId),
  layer_(layer),
  energy_(energy), 
  isSeed_(-1), 
  posxyz_(posx, posy, posz),
  axisxyz_(axisx, axisy, axisz) {
  
  posrep_.SetCoordinates( posxyz_.Rho(), 
			  posxyz_.Eta(), 
			  posxyz_.Phi() ); 

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( posxyz_ );    
  }  
}    


PFRecHit::PFRecHit(const PFRecHit& other) :
  detId_(other.detId_), 
  layer_(other.layer_), 
  energy_(other.energy_), 
  isSeed_(other.isSeed_),
  posxyz_(other.posxyz_), 
  posrep_(other.posrep_),
  axisxyz_(other.axisxyz_),
  cornersxyz_(other.cornersxyz_),
  neighboursIds4_(other.neighboursIds4_),
  neighboursIds8_(other.neighboursIds8_),
//   neighbours_(other.neighbours_),
  neighbours4_(other.neighbours4_),
  neighbours8_(other.neighbours8_) 
{}


PFRecHit::~PFRecHit() 
{}


const PFRecHit::REPPoint& PFRecHit::GetPositionREP() {
  if( posrep_ == REPPoint() ) {
    posrep_.SetCoordinates( posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi() );
  }
  return posrep_;
}


void PFRecHit::SetNeighbours( const vector<PFRecHit*>& neighbours ) {
  if( neighbours.size() != nNeighbours_ ) 
    throw cms::Exception("CellNeighbourVector") 
      << "number of neighbours must be nNeighbours_";
  
//   neighbours_.clear(); 
  neighboursIds4_.clear(); 
  neighboursIds8_.clear(); 
  neighbours4_.clear(); 
  neighbours8_.clear(); 
  
//   neighboursIds_.reserve(nNeighbours_); 
//   neighbours_.reserve(nNeighbours_); 

//   neighbours_ = neighbours;

  for(unsigned i=0; i<neighbours.size(); i++) {
    if( neighbours[i] ) {
      neighbours8_.push_back( neighbours[i] );      
      neighboursIds8_.push_back( neighbours[i]->GetDetId() );  
      if( !(i%2) ) {
	neighbours4_.push_back( neighbours[i] );
	neighboursIds4_.push_back( neighbours[i]->GetDetId() );  
      }
    }
  }
}


void PFRecHit::FindPtrsToNeighbours( const std::map<unsigned,  
				     reco::PFRecHit* >& allhits ) {

  neighbours4_.clear();
  neighbours8_.clear();
  
  typedef std::map<unsigned, reco::PFRecHit* >::const_iterator IDH;

  for(unsigned inid = 0; inid<neighboursIds8_.size(); inid++) {
    
    IDH ineighbour = allhits.find( neighboursIds8_[inid] );
    if( ineighbour != allhits.end() ) {
      neighbours8_.push_back( ineighbour->second );
    }
  }


  for(unsigned inid = 0; inid<neighboursIds4_.size(); inid++) {
    
    IDH ineighbour = allhits.find( neighboursIds4_[inid] );
    if( ineighbour != allhits.end() ) {
      neighbours4_.push_back( ineighbour->second );
    }
  }

}


void PFRecHit::SetNWCorner( double posx, double posy, double posz ) {
  SetCorner(0, posx, posy, posz);
}


void PFRecHit::SetSWCorner( double posx, double posy, double posz ) {
  SetCorner(1, posx, posy, posz);
}


void PFRecHit::SetSECorner( double posx, double posy, double posz ) {
  SetCorner(2, posx, posy, posz);
}


void PFRecHit::SetNECorner( double posx, double posy, double posz ) {
  SetCorner(3, posx, posy, posz);
}


void PFRecHit::SetCorner( unsigned i, double posx, double posy, double posz ) {
  assert( cornersxyz_.size() == nCorners_);
  assert( i<cornersxyz_.size() );

  cornersxyz_[i] = math::XYZPoint( posx, posy, posz);
}


ostream& reco::operator<<(ostream& out, const reco::PFRecHit& hit) {

  if(!out) return out;

//   reco::PFRecHit& nshit = const_cast<reco::PFRecHit& >(hit);
//   const reco::PFRecHit::REPPoint& posrep = nshit.GetPositionREP();
  
  const  math::XYZPoint& posxyz = hit.GetPositionXYZ();

  out<<"hit id:"<<hit.GetDetId()
     <<" layer:"<<hit.GetLayer()
     <<" energy:"<<hit.GetEnergy()
     <<" position: "
     <<" / "<<posxyz.Rho()<<","<<posxyz.Eta()<<","<<posxyz.Phi()
     <<" / "<<posxyz.X()<<","<<posxyz.Y()<<","<<posxyz.Z()
     <<" SEED: "<<hit.isSeed_<<endl;
  
//   out<<endl;
//   out<<"neighbours "<<endl;
//   for(unsigned i=0; i<hit.neighbours8_.size(); i++ ) {
//     out<<"\t"<< hit.neighbours8_[i]->GetDetId()<<endl;
//   }
//   out<<"--"<<endl;
//   for(unsigned i=0; i<hit.neighboursIds8_.size(); i++ ) {
//     out<<"\t"<< (hit.neighboursIds8_[i])<<endl;
//   }
  

  out<<"corners : "<<endl;
  const std::vector< math::XYZPoint >& corners = hit.GetCornersXYZ();
  for(unsigned i=0; i<corners.size(); i++) {
    out<<"\t"<<corners[i].X()<<","<<corners[i].Y()<<","<<corners[i].Z()<<endl;
  }
  
  return out;
}
