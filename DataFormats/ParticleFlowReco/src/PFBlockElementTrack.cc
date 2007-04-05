#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

#include <iostream>
#include <iomanip>

using namespace reco;
using namespace std;

void PFBlockElementTrack::Dump(ostream& out, 
			       const char* tab ) const {

  if(! out ) return;
  double charge = trackRef_->charge();
  double pt = trackRef_->innermostMeasurement()->momentum().Pt();
  double e  = trackRef_->innermostMeasurement()->momentum().E();
  

  out<<setprecision(3);
  out<<tab<<setw(10)<<"charge="<<setw(2)<<charge;
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  out<<", pT="<<setw(7)<<pt;
  out<<", E ="<<setw(7)<<e;
  out<<resetiosflags(ios::right|ios::fixed);
  // out<<resetiosflags(ios::fixed);
}
