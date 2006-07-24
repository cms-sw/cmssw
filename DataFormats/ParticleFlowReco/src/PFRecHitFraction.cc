#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
// #include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace reco;

// const unsigned    PFRecHit::nNeighbours_ = 8;
// const unsigned    PFRecHit::nCorners_ = 4;

// PFRecHit::PFRecHit(unsigned detId, int layer, 
// 		   double energy, 
// 		   double posx, double posy, double posz, 
// 		   double axisx, double axisy, double axisz) : 
//   detId_(detId), 
//   layer_(layer), 
//   energy_(energy), 
//   isSeed_(-1),
//   posxyz_(posx, posy, posz),
//   axisxyz_(axisx, axisy, axisz) {
  
//   posrep_.SetRho( posxyz_.Rho() );
//   posrep_.SetEta( posxyz_.Eta() );
//   posrep_.SetPhi( posxyz_.Phi() );

//   cornersrep_.reserve( nCorners_ );
//   for(unsigned i=0; i<nCorners_; i++) { 
//     cornersrep_.push_back( posrep_ );
//     cornersxyz_.push_back( posxyz_ );    
//   }
// }


// PFRecHit::PFRecHit(const PFRecHit& other) :
//   detId_(other.detId_), 
//   layer_(other.layer_), 
//   energy_(other.energy_), 
//   isSeed_(other.isSeed_),
//   posxyz_(other.posxyz_), 
//   posrep_(other.posrep_),
//   axisxyz_(other.axisxyz_),
//   cornersrep_(other.cornersrep_),
//   cornersxyz_(other.cornersxyz_) {
// }


// void PFRecHit::SetNeighbours( const vector<PFRecHit*>& neighbours ) {
//   if( neighbours.size() != nNeighbours_ ) 
//     throw cms::Exception("CellNeighbourVector") 
//       << "number of neighbours must be nNeighbours_";
  
//   neighbours_ = neighbours;

//   for(unsigned i=0; i<neighbours_.size(); i++) {
//     if( neighbours_[i] ) {
//       neighbours8_.push_back( neighbours_[i] );      
//       if( !(i%2) )
// 	neighbours4_.push_back( neighbours_[i] );
//     }
//   }
// }

// void PFRecHit::SetNWCorner( double posx, double posy, double posz ) {
//   SetCorner(0, posx, posy, posz);
// }

// void PFRecHit::SetSWCorner( double posx, double posy, double posz ) {
//   SetCorner(1, posx, posy, posz);
// }

// void PFRecHit::SetSECorner( double posx, double posy, double posz ) {
//   SetCorner(2, posx, posy, posz);
// }

// void PFRecHit::SetNECorner( double posx, double posy, double posz ) {
//   SetCorner(3, posx, posy, posz);
// }

// void PFRecHit::SetCorner( unsigned i, double posx, double posy, double posz ) {
//   assert( cornersrep_.size() == nCorners_);
//   assert( i<cornersrep_.size() );
  
//   math::XYZPoint cpos( posx, posy, posz);

//   cornersrep_[i].SetRho( cpos.Rho() );
//   cornersrep_[i].SetEta( cpos.Eta() );
//   cornersrep_[i].SetPhi( cpos.Phi() );

//   cornersxyz_[i] = cpos;
// }

// ostream& operator<<(ostream& out, const PFRecHit& hit) {

//   if(!out) return out;

//   const PFRecHit::REPPoint& posrep = hit.GetPositionREP();

//   out<<"hit id:"<<hit.GetDetId()
//      <<" layer:"<<hit.GetLayer()
//      <<" energy:"<<hit.GetEnergy()
//      <<" position: "
//      <<" / "<<posrep.Rho()<<","<<posrep.Eta()<<","<<posrep.Phi();
  
// //   out<<"corners : "<<endl;
// //   const std::vector< PFRecHit::REPPoint  >& corners = hit.GetCornersREP();
// //   for(unsigned i=0; i<corners.size(); i++) {
// //     out<<"\t"<<corners[i].Rho()<<","<<corners[i].Eta()<<","<<corners[i].Phi()<<endl;
// //   }
  
//   return out;
// }


ostream& reco::operator<<(std::ostream& out, 
		    const PFRecHitFraction& hit) {

  if(!out) return out;

  const reco::PFRecHit* rechit = hit.GetRecHit();
  out<<hit.GetEnergy()<<"\t"<<(*rechit);

  return out;
}
