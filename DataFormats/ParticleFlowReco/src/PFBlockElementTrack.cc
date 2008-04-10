#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/Common/interface/Ref.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iomanip>

using namespace reco;
using namespace std;


PFBlockElementTrack::PFBlockElementTrack(const PFRecTrackRef& ref, TrackType tracktype ) : 
  PFBlockElement( TRACK ),
  trackRefPF_( ref ), 
  trackRef_( ref->trackRef() ), 
  trackType_( tracktype ) {}


void PFBlockElementTrack::Dump(ostream& out, 
                               const char* tab ) const {
  
  if(! out ) return;
  
  if( !trackRef_.isNull() ) {
    
    double charge = trackRef_->charge();
    double pt = trackRef_->pt();
    double p = trackRef_->p();
    string s = "  at vertex";
    double tracketa = trackRef_->eta();
    double trackphi = trackRef_->phi();
    const reco::PFTrajectoryPoint& atECAL 
      = trackRefPF_->extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax );
    // check if  reach ecal Shower max 
    if( atECAL.isValid() ) { 
      s = "  at ECAL shower max";  
      tracketa = atECAL.positionXYZ().Eta();
      trackphi = atECAL.positionXYZ().Phi();
    }
    
    out<<setprecision(0);
    out<<tab<<setw(7)<<"charge="<<setw(3)<<charge;
    out<<setprecision(3);
    out<<setiosflags(ios::right);
    out<<setiosflags(ios::fixed);
    out<<", pT ="<<setw(7)<<pt;
    out<<", p ="<<setw(7)<<p;
    out<<" (eta,phi)= (";
    out<<tracketa<<",";
    out<<trackphi<<")" << s;
    
    out<<resetiosflags(ios::right|ios::fixed);  }
}
