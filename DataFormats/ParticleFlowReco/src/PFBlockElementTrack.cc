#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <iostream>
#include <iomanip>

using namespace reco;
using namespace std;


PFBlockElementTrack::PFBlockElementTrack(const PFRecTrackRef& ref ) : 
  PFBlockElement( TRACK ),
  trackRefPF_( ref ), 
  trackRef_( ref->trackRef() ) {}


void PFBlockElementTrack::Dump(ostream& out, 
			       const char* tab ) const {

  if(! out ) return;

  
//   if( !trackRefPF_.isNull() ) {
//     double charge = trackRef_->charge();
//     double pt = trackRef_->innermostMeasurement()->momentum().Pt();
//     double e  = trackRef_->innermostMeasurement()->momentum().E();
  
    
//     out<<setprecision(3);
//     out<<tab<<setw(10)<<"charge="<<setw(2)<<charge;
//     out<<setiosflags(ios::right);
//     out<<setiosflags(ios::fixed);
//     out<<", pT="<<setw(7)<<pt;
//     out<<", E ="<<setw(7)<<e;
//     out<<resetiosflags(ios::right|ios::fixed);
//     // out<<resetiosflags(ios::fixed);
//   }
  if( !trackRef_.isNull() ) {
    double charge = trackRef_->charge();
    double pt = trackRef_->pt();
    double p = trackRef_->p();
    
    
    out<<setprecision(3);
    out<<tab<<setw(10)<<"charge="<<setw(2)<<charge;
    out<<setiosflags(ios::right);
    out<<setiosflags(ios::fixed);
    out<<", pT="<<setw(7)<<pt;
    out<<", p ="<<setw(7)<<p;
    out<<resetiosflags(ios::right|ios::fixed);
    // out<<resetiosflags(ios::fixed);
  }
}
