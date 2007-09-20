#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include <iomanip>

using namespace reco;
using namespace std;


PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


// reco::TrackRef PFCandidate::trackRef() const {

//   const edm::OwnVector< reco::PFBlockElement >& elements
//     = blockRef()->elements();
  
//   unsigned ntracks = 0;
//   unsigned itrack = 0;
//   for(unsigned i=0; i<elements.size(); i++ ) {
//     if( elements[i].type() == PFBlockElement::TRACK) {
//       itrack = i;
//       ntracks++;
//     }
//   }
//   assert(ntracks<2);
  
//   if(ntracks == 1) 
//     return elements[itrack].trackRef();
//   else 
//     return reco::TrackRef();
// }


ostream& reco::operator<<(ostream& out, 
			  const PFCandidate& c ) {
  
  if(!out) return out;
  
  out<<"\tPFCandidate type: "<<c.particleId();
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  out<<setprecision(3);
  out<<" ( pT="<<setw(7)<<c.pt();
  out<<", E ="<<setw(7)<<c.energy()<<" ) ";
  out<<", iele = unknown";
  
  //   for(unsigned i=0; i<c.elementIndices_.size(); i++) {
  //     out<<c.elementIndices_[0]<<" ";
  //   }
  //   out<<endl;
  
  out<<resetiosflags(ios::right|ios::fixed);
  
  //  out<< *(c.blockRef_)<<endl;
  
  return out;
}

